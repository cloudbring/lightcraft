//! Crash-safe writes of user-visible files: exports, renders, DNG conversions, smart previews and
//! XMP sidecars all go through here (the catalog's own files use [`crate::store::FsStore`]).
//!
//! **Write policy** (issue #134): what can't be recreated — originals, XMP sidecars, the catalog,
//! settings, DNGs that replace raws, smart previews (the only stand-in while an original is
//! offline), an edit copy that becomes a library photo — is written **durably**: synced to disk
//! before it counts as written. What can always be made again from the originals — exports,
//! renders, screenshots, merge previews — is written **atomically** only: a half-written file
//! never replaces a good one, but there is no per-file sync (on a USB drive or a NAS that sync
//! dominates a large export).
//!
//! - [`write_atomic`] replaces a file so that after a crash, a full disk or an unplugged drive
//!   either the old or the complete new content is there — never a truncated mix. The content
//!   goes to a new temp file in the same folder (unique name, `create_new`), is synced to disk and
//!   renamed over the target; then the folder is synced (POSIX). Any failure removes the temp
//!   file and leaves the target as it was.
//! - [`write_atomic_nosync`] is the same temp file + rename without the two syncs, for
//!   recreatable outputs only: a failed write still removes the temp file and leaves the target as
//!   it was, but after a power cut the new file may be lost (on some file systems left empty) —
//!   it is then simply exported again.
//! - [`write_new`] / [`write_new_unique`] write a file that must not exist yet: the same temp file
//!   and sync, plus a read-back check, and the file then appears under its final name **without
//!   replacing anything** (a hard link, which fails on a taken name; where the volume has no hard
//!   links, an exclusive create and copy, which can be visible before the copy is complete).
//! - [`same_file`] tells whether two paths name the same file on disk (device + inode on Unix,
//!   volume serial number + file index on Windows,
//!   resolved paths elsewhere), for "is this target an original?" checks.

use std::cell::{Cell, RefCell};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

thread_local! {
    static FAIL_AFTER: Cell<Option<u64>> = const { Cell::new(None) };
    static SYNCS: Cell<u64> = const { Cell::new(0) };
}

/// How many file and folder syncs this module made **on the calling thread** (for tests: a
/// durable write path syncs, a recreatable output doesn't).
#[doc(hidden)]
pub fn syncs_on_this_thread() -> u64 {
    SYNCS.with(Cell::get)
}

fn sync_file(f: &File) -> io::Result<()> {
    SYNCS.with(|c| c.set(c.get() + 1));
    f.sync_all()
}

/// Fault injection for tests (in any crate): while the returned guard lives, every write made by
/// this module **on the calling thread** fails once `bytes` bytes of a file were written — as a
/// full disk or an unplugged drive would.
#[doc(hidden)]
pub fn fail_writes_after(bytes: u64) -> FaultGuard {
    FAIL_AFTER.with(|c| c.set(Some(bytes)));
    FaultGuard(())
}

/// Ends a [`fail_writes_after`] when dropped.
#[doc(hidden)]
pub struct FaultGuard(());

impl Drop for FaultGuard {
    fn drop(&mut self) {
        FAIL_AFTER.with(|c| c.set(None));
    }
}

/// Counts what is written; fails past an injected limit.
struct Limited<W> {
    inner: W,
    n: u64,
    limit: Option<u64>,
}

impl<W: Write> Write for Limited<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let buf = match self.limit {
            Some(l) => {
                let room = usize::try_from(l.saturating_sub(self.n)).unwrap_or(usize::MAX);
                if room == 0 && !buf.is_empty() {
                    return Err(io::Error::other("write failed (injected fault)"));
                }
                buf.get(..room.min(buf.len())).unwrap_or(buf)
            }
            None => buf,
        };
        let k = self.inner.write(buf)?;
        self.n += k as u64;
        Ok(k)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Make a rename or a new name in `dir` durable (POSIX). Opening a directory fails on Windows,
/// where NTFS journals the rename itself, so errors are ignored.
pub fn sync_dir(dir: &Path) {
    if let Ok(d) = File::open(dir) {
        let _ = sync_file(&d);
    }
}

/// Replace (or create) `path` with `data` atomically and durably; see the module docs. The
/// folder must exist.
pub fn write_atomic(path: &Path, data: &[u8]) -> io::Result<()> {
    write_atomic_with(path, &mut |w| w.write_all(data)).map(|_| ())
}

/// [`write_atomic`] with streamed content: `fill` writes it (an error aborts and leaves the old
/// file). Returns the length written.
pub fn write_atomic_with(path: &Path, fill: &mut dyn FnMut(&mut dyn Write) -> io::Result<()>) -> io::Result<u64> {
    replace_with(path, true, fill)
}

/// Replace (or create) `path` with `data` atomically but **without syncing** it to disk — only for
/// outputs that can always be recreated (exports, renders, screenshots); see the module docs.
/// Anything that can't be made again goes through [`write_atomic`].
pub fn write_atomic_nosync(path: &Path, data: &[u8]) -> io::Result<()> {
    replace_with(path, false, &mut |w| w.write_all(data)).map(|_| ())
}

/// Temp file (synced when `sync`) renamed over `path`, then the folder synced when `sync`.
fn replace_with(path: &Path, sync: bool, fill: &mut dyn FnMut(&mut dyn Write) -> io::Result<()>) -> io::Result<u64> {
    let filesystem = ReplacementFs { sync, stage: RefCell::new(None) };
    replace_with_fs(&filesystem, path, fill)
}

fn replace_with_fs(filesystem: &dyn craft_atomic::AtomicFs, path: &Path, fill: &mut dyn FnMut(&mut dyn Write) -> io::Result<()>) -> io::Result<u64> {
    let mut n = 0;
    craft_atomic::WriteOptions::default().atomic_write_with_custom(filesystem, craft_atomic::RenameRetry::platform(), path, |file| {
        let limit = FAIL_AFTER.with(Cell::get);
        let mut writer = Limited { inner: io::BufWriter::with_capacity(1 << 20, file), n: 0, limit };
        fill(&mut writer)?;
        writer.flush()?;
        n = writer.n;
        Ok(())
    })?;
    Ok(n)
}

struct ReplacementFs {
    sync: bool,
    stage: RefCell<Option<PathBuf>>,
}

impl craft_atomic::AtomicFs for ReplacementFs {
    fn create_new(&self, path: &Path, permissions: craft_atomic::NewFilePermissions) -> io::Result<File> {
        let file = craft_atomic::AtomicFs::create_new(&craft_atomic::RealFs, path, permissions)?;
        if self.stage.borrow().is_none() {
            *self.stage.borrow_mut() = Some(path.to_path_buf());
        }
        Ok(file)
    }
    fn sync(&self, file: &File) -> io::Result<()> {
        if self.sync {
            SYNCS.with(|count| count.set(count.get() + 1));
            craft_atomic::AtomicFs::sync(&craft_atomic::RealFs, file)
        } else {
            Ok(())
        }
    }
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        craft_atomic::AtomicFs::rename(&craft_atomic::RealFs, from, to)
    }
    fn hard_link(&self, from: &Path, to: &Path) -> io::Result<()> {
        craft_atomic::AtomicFs::hard_link(&craft_atomic::RealFs, from, to)
    }
    fn remove(&self, path: &Path) -> io::Result<()> {
        craft_atomic::AtomicFs::remove(&craft_atomic::RealFs, path)
    }
    fn sync_dir(&self, dir: &Path) -> io::Result<()> {
        if self.sync {
            SYNCS.with(|count| count.set(count.get() + 1));
            craft_atomic::AtomicFs::sync_dir(&craft_atomic::RealFs, dir)
        } else {
            Ok(())
        }
    }
    fn set_permissions(&self, file: &File, permissions: fs::Permissions) -> io::Result<()> {
        craft_atomic::AtomicFs::set_permissions(&craft_atomic::RealFs, file, permissions)
    }
}

/// Write `data` as the new file `path`: never replaces an existing file (`AlreadyExists`), and
/// the staged file is synced and read back before publication. On volumes without hard links,
/// the exclusive-copy fallback may expose a partially copied file during publication.
pub fn write_new(path: &Path, data: &[u8]) -> io::Result<()> {
    write_new_unique(&mut std::iter::once(path.to_path_buf()), data).map(|_| ())
}

/// [`write_new`] under the first free name of `candidates` (all in the same folder, e.g.
/// `IMG_1.dng`, `IMG_1-2.dng`, …): attempts each name until one is available, without replacing
/// another file. Returns that name; `AlreadyExists` when every candidate is taken.
pub fn write_new_unique(candidates: &mut dyn Iterator<Item = PathBuf>, data: &[u8]) -> io::Result<PathBuf> {
    let mut first = None;
    for path in candidates.take(100_000) {
        if first.is_none() {
            first = Some(path.clone());
        }
        let filesystem = ReplacementFs { sync: true, stage: RefCell::new(None) };
        let result = craft_atomic::WriteOptions::default().write_new_with_custom(&filesystem, &path, |file| {
            let mut writer = Limited { inner: &mut *file, n: 0, limit: FAIL_AFTER.with(Cell::get) };
            writer.write_all(data)?;
            writer.flush()?;
            craft_atomic::AtomicFs::sync(&filesystem, file)?;
            let stage = filesystem.stage.borrow();
            let stage = stage.as_deref().ok_or_else(|| io::Error::other("missing staging file"))?;
            if !same_content(stage, data)? {
                return Err(io::Error::other("the written file reads back different (failing drive or connection?)"));
            }
            Ok(())
        });
        match result {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    match first {
        Some(path) => Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{}: no free file name", path.display()))),
        None => Err(io::Error::new(io::ErrorKind::InvalidInput, "no file name to write")),
    }
}

/// Does the file at `path` hold exactly `data`?
fn same_content(path: &Path, data: &[u8]) -> io::Result<bool> {
    use std::io::Read;
    let mut f = File::open(path)?;
    if f.metadata()?.len() != data.len() as u64 {
        return Ok(false);
    }
    let mut buf = vec![0u8; 1 << 20];
    let mut at = 0usize;
    loop {
        let k = match f.read(&mut buf) {
            Ok(0) => return Ok(at == data.len()),
            Ok(k) => k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if data.get(at..at + k) != buf.get(..k) {
            return Ok(false);
        }
        at += k;
    }
}

/// Do `a` and `b` name the same existing file (through hard links, symlinks, `..`,
/// case-insensitive names)?
pub fn same_file(a: &Path, b: &Path) -> bool {
    compare_files(a, b, false).unwrap_or(false)
}

/// [`same_file`] for a check that must not let an unknown identity pass: `Err` when the metadata
/// of either path cannot be read for a reason other than the path not existing (for example a
/// folder it cannot enter), or when both paths exist with the same length but the file identity
/// of either cannot be read (on Windows, when either file cannot be opened or reports file index 0). A path that does not
/// exist is still `Ok(false)`.
pub fn same_file_strict(a: &Path, b: &Path) -> io::Result<bool> {
    compare_files(a, b, true)
}

fn compare_files(a: &Path, b: &Path, strict: bool) -> io::Result<bool> {
    let missing = |e: &io::Error| matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory);
    let (ma, mb) = match (fs::metadata(a), fs::metadata(b)) {
        (Ok(ma), Ok(mb)) => (ma, mb),
        (Err(e), _) | (_, Err(e)) if strict && !missing(&e) => return Err(e),
        _ => return Ok(false),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(ma.dev() == mb.dev() && ma.ino() == mb.ino())
    }
    #[cfg(not(unix))]
    {
        if ma.len() != mb.len() {
            return Ok(false);
        }
        // A hard link has its own path but shares the volume serial number and file index with
        // its original. A file index of 0 produces an error in strict mode. Lenient mode compares resolved paths.
        // On ReFS two files can share the 64-bit file index, so the lengths are compared first.
        #[cfg(windows)]
        {
            match file_ids(a, b).and_then(|(ia, ib)| same_file_id(ia, ib, strict)) {
                Ok(true) => return Ok(true),
                Ok(_) => {}
                Err(e) if strict => return Err(e),
                Err(_) => {}
            }
        }
        Ok(match (fs::canonicalize(a), fs::canonicalize(b)) {
            (Ok(x), Ok(y)) => x == y || x.to_string_lossy().to_lowercase() == y.to_string_lossy().to_lowercase(),
            _ => false,
        })
    }
}

/// The volume serial number and file index of `a` and `b`, read while both files are open.
#[cfg(windows)]
fn file_ids(a: &Path, b: &Path) -> io::Result<((u64, u64), (u64, u64))> {
    use winapi_util::{Handle, file::information};
    let (ha, hb) = (Handle::from_path_any(a)?, Handle::from_path_any(b)?);
    let (ia, ib) = (information(&ha)?, information(&hb)?);
    Ok(((ia.volume_serial_number(), ia.file_index()), (ib.volume_serial_number(), ib.file_index())))
}

/// The comparison uses each file's volume serial number and file index. Strict mode returns an error for file index 0.
#[cfg(any(windows, test))]
fn same_file_id(a: (u64, u64), b: (u64, u64), strict: bool) -> io::Result<bool> {
    if strict && (a.1 == 0 || b.1 == 0) {
        return Err(io::Error::other("file identity is unavailable"));
    }
    Ok(a.1 != 0 && b.1 != 0 && a == b)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FaultFs(&'static str);

    impl craft_atomic::AtomicFs for FaultFs {
        fn create_new(&self, path: &Path, permissions: craft_atomic::NewFilePermissions) -> io::Result<File> {
            craft_atomic::AtomicFs::create_new(&craft_atomic::RealFs, path, permissions)
        }
        fn sync(&self, file: &File) -> io::Result<()> {
            if self.0 == "sync" { Err(io::Error::other("injected sync failure")) } else { craft_atomic::AtomicFs::sync(&craft_atomic::RealFs, file) }
        }
        fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
            if self.0 == "rename" {
                Err(io::Error::other("injected rename failure"))
            } else {
                craft_atomic::AtomicFs::rename(&craft_atomic::RealFs, from, to)
            }
        }
        fn hard_link(&self, from: &Path, to: &Path) -> io::Result<()> {
            if self.0 == "race" {
                fs::write(to, b"other writer")?;
                return Err(io::Error::other("hard links unavailable"));
            }
            craft_atomic::AtomicFs::hard_link(&craft_atomic::RealFs, from, to)
        }
        fn remove(&self, path: &Path) -> io::Result<()> {
            craft_atomic::AtomicFs::remove(&craft_atomic::RealFs, path)
        }
        fn sync_dir(&self, dir: &Path) -> io::Result<()> {
            if self.0 == "directory" {
                Err(io::Error::other("injected directory failure"))
            } else {
                craft_atomic::AtomicFs::sync_dir(&craft_atomic::RealFs, dir)
            }
        }
        fn set_permissions(&self, file: &File, permissions: fs::Permissions) -> io::Result<()> {
            craft_atomic::AtomicFs::set_permissions(&craft_atomic::RealFs, file, permissions)
        }
    }

    #[test]
    fn interrupted_replacements_never_publish_torn_content() {
        let dir = temp("fault-points");
        let path = dir.join("photo.jpg");
        for fault in ["sync", "rename", "directory", "none"] {
            fs::write(&path, b"old content").unwrap();
            let result = replace_with_fs(&FaultFs(fault), &path, &mut |writer| writer.write_all(b"complete new content"));
            let actual = fs::read(&path).unwrap();
            assert!(actual == b"old content" || actual == b"complete new content", "{fault}");
            if fault == "sync" || fault == "rename" {
                assert!(result.is_err());
                assert_eq!(actual, b"old content");
            } else {
                assert!(result.is_ok());
                assert_eq!(actual, b"complete new content");
            }
        }
        let _ = fs::remove_dir_all(dir);
    }

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("lc-safe-file-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect();
        v.sort();
        v
    }

    #[test]
    fn fallback_publish_never_replaces_a_file_that_appears_at_the_same_time() {
        let d = temp("nolink-race");
        let dest = d.join("IMG_1.dng");
        let e = craft_atomic::WriteOptions::default().write_new_custom(&FaultFs("race"), &dest, b"new").unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(&dest).unwrap(), b"other writer");
        fs::remove_file(&dest).unwrap();
        write_new(&dest, b"new").unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"new");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn atomic_write_replaces_and_leaves_no_temp() {
        let d = temp("replace");
        let p = d.join("out.jpg");
        write_atomic(&p, b"one").unwrap();
        write_atomic(&p, b"two two").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"two two");
        assert_eq!(names(&d), vec!["out.jpg"]);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_failure_mid_write_keeps_the_old_file() {
        let d = temp("midwrite");
        let p = d.join("photo.jpg");
        fs::write(&p, b"the original bytes").unwrap();
        {
            let _f = fail_writes_after(4);
            let e = write_atomic(&p, &[7u8; 10_000]).unwrap_err();
            assert!(e.to_string().contains("injected"), "{e}");
            assert!(write_new(&d.join("new.dng"), &[1u8; 100]).is_err());
        }
        assert_eq!(fs::read(&p).unwrap(), b"the original bytes", "the target is untouched");
        assert_eq!(names(&d), vec!["photo.jpg"], "no temp file and no partial new file");
        // a fill error aborts the same way
        let r = write_atomic_with(&p, &mut |w| {
            w.write_all(b"half")?;
            Err(io::Error::other("encoder failed"))
        });
        assert!(r.is_err());
        assert_eq!(fs::read(&p).unwrap(), b"the original bytes");
        assert_eq!(names(&d), vec!["photo.jpg"]);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn recreatable_outputs_are_atomic_without_a_sync() {
        let d = temp("nosync");
        let p = d.join("export.jpg");
        let before = syncs_on_this_thread();
        write_atomic_nosync(&p, b"one").unwrap();
        write_atomic_nosync(&p, b"two two").unwrap();
        assert_eq!(syncs_on_this_thread(), before, "no file or folder sync");
        assert_eq!(fs::read(&p).unwrap(), b"two two");
        assert_eq!(names(&d), vec!["export.jpg"], "no temp file left");
        // still atomic: a failure part-way leaves the previous file and no temp file
        {
            let _f = fail_writes_after(3);
            assert!(write_atomic_nosync(&p, &[9u8; 10_000]).is_err());
        }
        assert_eq!(fs::read(&p).unwrap(), b"two two");
        assert_eq!(names(&d), vec!["export.jpg"]);
        // the durable writers do sync (the file; the folder too where it can be opened)
        let before = syncs_on_this_thread();
        write_atomic(&p, b"three").unwrap();
        let mid = syncs_on_this_thread();
        assert!(mid > before);
        write_new(&d.join("new.dng"), b"dng").unwrap();
        assert!(syncs_on_this_thread() > mid);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn new_files_never_replace_anything() {
        let d = temp("new");
        let p = d.join("a.dng");
        fs::write(&p, b"keep").unwrap();
        assert_eq!(write_new(&p, b"other").unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(&p).unwrap(), b"keep");
        let mut c = (1..).map(|k| if k == 1 { d.join("a.dng") } else { d.join(format!("a-{k}.dng")) });
        let got = write_new_unique(&mut c, b"fresh").unwrap();
        assert_eq!(got, d.join("a-2.dng"));
        assert_eq!(fs::read(&got).unwrap(), b"fresh");
        assert_eq!(names(&d), vec!["a-2.dng", "a.dng"], "the temp file is gone");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn same_file_sees_a_hard_link_under_another_name() {
        let d = temp("same-link");
        let p = d.join("IMG_1.jpg");
        fs::write(&p, b"original").unwrap();
        let linked = d.join("linked.png");
        match fs::hard_link(&p, &linked) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => panic!("{}: {e}", linked.display()),
            // no hard links on this volume (FAT/exFAT, some shares), as in `publish_new`
            Err(e) => {
                let _ = fs::remove_dir_all(&d);
                lightcraft_testkit::skip!(System, "no hard links in {}: {e}", d.display());
                return;
            }
        }
        fs::write(d.join("copy.png"), b"original").unwrap();
        assert!(same_file(&p, &linked), "a hard link is the same file");
        assert!(same_file(&linked, &p));
        assert!(!same_file(&p, &d.join("copy.png")), "a copy with the same content and length is another file");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn same_file_sees_through_other_spellings() {
        let d = temp("same");
        let p = d.join("x.jpg");
        fs::write(&p, b"x").unwrap();
        fs::write(d.join("y.jpg"), b"x").unwrap();
        fs::create_dir_all(d.join("sub")).unwrap();
        assert!(same_file(&p, &d.join("sub/../x.jpg")));
        assert!(!same_file(&p, &d.join("y.jpg")));
        assert!(!same_file(&p, &d.join("missing.jpg")));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn strict_comparison_reports_a_path_it_cannot_query() {
        let d = temp("strict-query");
        let p = d.join("x.jpg");
        fs::write(&p, b"x").unwrap();
        // A NUL byte makes the metadata lookup fail with an error other than "not found" on
        // every platform, as a folder the process cannot enter does.
        let unreadable = Path::new("IMG_1\0.jpg");
        assert!(same_file_strict(unreadable, &p).is_err());
        assert!(same_file_strict(&p, unreadable).is_err());
        assert!(!same_file(unreadable, &p), "the lenient comparison still answers false");
        assert!(!same_file_strict(&d.join("missing.jpg"), &p).unwrap(), "a missing original is another file");
        assert!(!same_file_strict(&p.join("under-a-file.jpg"), &p).unwrap());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn a_zero_file_index_is_not_a_file_identity() {
        for strict in [false, true] {
            assert!(same_file_id((7, 42), (7, 42), strict).unwrap());
            assert!(!same_file_id((7, 42), (7, 43), strict).unwrap());
            assert!(!same_file_id((7, 42), (8, 42), strict).unwrap());
        }
        for (a, b) in [((7, 0), (7, 0)), ((0, 0), (0, 0)), ((7, 42), (7, 0)), ((7, 0), (7, 42))] {
            assert!(same_file_id(a, b, true).is_err(), "strict comparison must reject an unavailable identity: {a:?}, {b:?}");
            assert!(!same_file_id(a, b, false).unwrap(), "lenient comparison can fall back to resolved paths");
        }
    }
}
