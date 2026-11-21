use rug::Float as RugFloat;
use std::cell::RefCell;

pub(crate) const MAX_ARENA_LEN: usize = 1usize << 30;

thread_local! {
    static ARENA: RefCell<Vec<RugFloat>> = const { RefCell::new(Vec::new()) };
    static ARENA_GEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[inline]
pub(crate) fn push(value: RugFloat) -> u32 {
    ARENA.with(|cell| {
        let mut a = cell.borrow_mut();
        let idx = a.len();
        if idx >= MAX_ARENA_LEN {
            panic!("MpfrFloat arena exhausted.");
        }
        a.push(value);
        idx as u32
    })
}

#[inline]
pub(crate) fn get(handle: u32) -> RugFloat {
    ARENA.with(|cell| cell.borrow()[handle as usize].clone())
}

#[inline]
pub(crate) fn with<R>(handle: u32, f: impl FnOnce(&RugFloat) -> R) -> R {
    ARENA.with(|cell| f(&cell.borrow()[handle as usize]))
}

#[inline]
pub(crate) fn with2<R>(a: u32, b: u32, f: impl FnOnce(&RugFloat, &RugFloat) -> R) -> R {
    ARENA.with(|cell| {
        let arena = cell.borrow();
        f(&arena[a as usize], &arena[b as usize])
    })
}

pub fn reset_arena() {
    ARENA.with(|cell| cell.borrow_mut().clear());
    ARENA_GEN.with(|g| g.set(g.get().wrapping_add(1)));
}

pub(crate) fn arena_generation() -> u64 { ARENA_GEN.with(|g| g.get()) }
pub fn arena_len() -> usize { ARENA.with(|cell| cell.borrow().len()) }

#[inline]
pub(crate) fn push_zero() -> u32 {
    push(RugFloat::with_val(super::default_precision(), 0))
}

#[inline]
pub(crate) fn push_one() -> u32 {
    push(RugFloat::with_val(super::default_precision(), 1))
}
