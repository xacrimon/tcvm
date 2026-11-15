//! The emitter's state at a point, as one change-log table: a block's end
//! state is sealed into a snapshot, and a block starts from its
//! predecessors' snapshots at a cost proportional to what they changed
//! (Turboshaft's `SnapshotTable` is the model).

/// A key's value where nothing is known.
pub(crate) const NONE: u32 = u32::MAX;
/// No value is this: a predecessor that has not written a key, while a
/// merge is collected.
const UNSET: u32 = u32::MAX - 1;

const KINDS: usize = 2;

/// A key: its kind and an index dense within the kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Key(u32);

impl Key {
    /// A bytecode register: the value it holds.
    pub(crate) fn reg(r: u32) -> Key {
        Key(r)
    }

    /// A value: its fact's bits where refined.
    pub(crate) fn fact(v: u32) -> Key {
        Key(1 << 31 | v)
    }

    pub(crate) fn is_fact(self) -> bool {
        self.0 >> 31 != 0
    }

    pub(crate) fn index(self) -> u32 {
        self.0 & !(1 << 31)
    }
}

/// A sealed state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Snap(u32);

impl Snap {
    /// The state before anything was set.
    pub(crate) const ROOT: Snap = Snap(NONE);
}

struct SnapData {
    parent: Snap,
    depth: u32,
    /// Its changes from its parent, `log[start..end]`.
    start: u32,
    end: u32,
}

struct Change {
    key: Key,
    old: u32,
    new: u32,
}

#[derive(Default)]
pub(crate) struct Table {
    cur: [Vec<u32>; KINDS],
    log: Vec<Change>,
    snaps: Vec<SnapData>,
    /// The snapshot `cur` is at, less the open changes `log[open..]`.
    at: Snap,
    open: u32,
    /// The keys the predecessors of the last `start` disagree on, with each
    /// predecessor's value: `merge_vals[i * np + k]`.
    merge_keys: Vec<Key>,
    merge_vals: Vec<u32>,
    np: usize,
    /// A key's index in `merge_keys`, plus one.
    mark: [Vec<u32>; KINDS],
    path: Vec<Snap>,
}

impl Default for Snap {
    fn default() -> Snap {
        Snap::ROOT
    }
}

impl Table {
    pub(crate) fn reset(&mut self) {
        for k in 0..KINDS {
            self.cur[k].clear();
            self.mark[k].clear();
        }
        self.log.clear();
        self.snaps.clear();
        self.at = Snap::ROOT;
        self.open = 0;
        self.merge_keys.clear();
        self.merge_vals.clear();
    }

    #[inline]
    pub(crate) fn get(&self, key: Key) -> u32 {
        let c = &self.cur[key.is_fact() as usize];
        c.get(key.index() as usize).copied().unwrap_or(NONE)
    }

    fn put(&mut self, key: Key, v: u32) {
        let c = &mut self.cur[key.is_fact() as usize];
        let i = key.index() as usize;
        if i >= c.len() {
            c.resize(i + 1, NONE);
        }
        c[i] = v;
    }

    pub(crate) fn set(&mut self, key: Key, new: u32) {
        debug_assert!(new != UNSET);
        let old = self.get(key);
        if old != new {
            self.put(key, new);
            self.log.push(Change { key, old, new });
        }
    }

    /// Seal the changes since the last `start`.
    pub(crate) fn seal(&mut self) -> Snap {
        let depth = self.depth(self.at) + 1;
        self.snaps.push(SnapData {
            parent: self.at,
            depth,
            start: self.open,
            end: self.log.len() as u32,
        });
        self.open = self.log.len() as u32;
        self.at = Snap(self.snaps.len() as u32 - 1);
        self.at
    }

    fn depth(&self, s: Snap) -> u32 {
        if s == Snap::ROOT {
            0
        } else {
            self.snaps[s.0 as usize].depth
        }
    }

    fn parent(&self, s: Snap) -> Snap {
        self.snaps[s.0 as usize].parent
    }

    fn common(&self, mut a: Snap, mut b: Snap) -> Snap {
        while a != b {
            if self.depth(a) >= self.depth(b) {
                a = self.parent(a);
            } else {
                b = self.parent(b);
            }
        }
        a
    }

    fn move_to(&mut self, to: Snap) {
        let mut b = to;
        self.path.clear();
        while self.at != b {
            if self.depth(self.at) >= self.depth(b) {
                let s = &self.snaps[self.at.0 as usize];
                let (range, parent) = (s.start as usize..s.end as usize, s.parent);
                for k in range.rev() {
                    let Change { key, old, .. } = self.log[k];
                    self.put(key, old);
                }
                self.at = parent;
            } else {
                self.path.push(b);
                b = self.parent(b);
            }
        }
        while let Some(s) = self.path.pop() {
            let s = &self.snaps[s.0 as usize];
            for k in s.start as usize..s.end as usize {
                let Change { key, new, .. } = self.log[k];
                self.put(key, new);
            }
        }
        self.at = to;
    }

    /// Start from the predecessors' states: what they agree on stays, and
    /// what any of them changed since their common ancestor is listed
    /// (`merges`, `pred_value`) with the ancestor's value current, for the
    /// caller to `set`. The previous state must be sealed.
    pub(crate) fn start(&mut self, preds: &[Snap]) {
        debug_assert!(self.open as usize == self.log.len());
        for &key in &self.merge_keys {
            self.mark[key.is_fact() as usize][key.index() as usize] = 0;
        }
        self.merge_keys.clear();
        self.merge_vals.clear();
        self.np = preds.len();
        let Some((&first, rest)) = preds.split_first() else {
            return self.move_to(Snap::ROOT);
        };
        let base = rest.iter().fold(first, |a, &b| self.common(a, b));
        self.move_to(base);
        if rest.is_empty() {
            return;
        }
        for (k, &p) in preds.iter().enumerate() {
            let mut s = p;
            while s != base {
                let d = &self.snaps[s.0 as usize];
                let (range, parent) = (d.start as usize..d.end as usize, d.parent);
                // Backward: a key's first change met is its value at `p`.
                for at in range.rev() {
                    let Change { key, new, .. } = self.log[at];
                    let m = &mut self.mark[key.is_fact() as usize];
                    let i = key.index() as usize;
                    if i >= m.len() {
                        m.resize(i + 1, 0);
                    }
                    if m[i] == 0 {
                        self.merge_keys.push(key);
                        m[i] = self.merge_keys.len() as u32;
                        self.merge_vals
                            .resize(self.merge_vals.len() + self.np, UNSET);
                    }
                    let cell = &mut self.merge_vals[(m[i] as usize - 1) * self.np + k];
                    if *cell == UNSET {
                        *cell = new;
                    }
                }
                s = parent;
            }
        }
        for i in 0..self.merge_keys.len() {
            let base = self.get(self.merge_keys[i]);
            for cell in &mut self.merge_vals[i * self.np..(i + 1) * self.np] {
                if *cell == UNSET {
                    *cell = base;
                }
            }
        }
    }

    pub(crate) fn merges(&self) -> usize {
        self.merge_keys.len()
    }

    /// Merge candidate `i`: its key and its value at each predecessor.
    pub(crate) fn merge(&self, i: usize) -> (Key, &[u32]) {
        (
            self.merge_keys[i],
            &self.merge_vals[i * self.np..(i + 1) * self.np],
        )
    }

    /// A key's value at the end of predecessor `k` of the last `start`.
    pub(crate) fn pred_value(&self, key: Key, k: usize) -> u32 {
        let m = &self.mark[key.is_fact() as usize];
        match m.get(key.index() as usize) {
            Some(&i) if i != 0 => self.merge_vals[(i as usize - 1) * self.np + k],
            // Unchanged since the common ancestor by every predecessor.
            _ => self.get(key),
        }
    }
}
