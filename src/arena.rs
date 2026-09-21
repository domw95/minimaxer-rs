//! Storage for search nodes.
//!
//! `Node` used to own its children inline (`Vec<(M, Node)>`), which made the
//! tree a web of separate allocations: one per node for its children vector,
//! one for its move list, each with its own malloc header and whatever
//! capacity slack doubling left behind. Here the nodes live in the arena and a
//! child is a [`NodeId`] into it.
//!
//! The reason that matters is not the bytes it saves directly -- it is that
//! discarding the parts of the tree a deepening pass no longer needs becomes
//! "copy what is kept into a fresh arena and drop the old one". That is
//! O(kept) rather than O(discarded), and because the arena hands whole chunks
//! back to the allocator it actually returns memory to the OS, which walking a
//! pointer tree freeing 248-byte nodes into a free list does not.

use crate::node::Node;

/// Index of a node in an [`Arena`].
pub type NodeId = u32;

/// Nodes per chunk.
///
/// Chunks are allocated at full capacity and never grow, so a node never moves
/// once allocated: no reallocation copy, and no transient 1.5x spike from
/// doubling one enormous vector. At Azul's node size a chunk is about 1 MiB,
/// which glibc services with `mmap` and returns to the OS on free.
const CHUNK_BITS: u32 = 12;
const CHUNK_LEN: usize = 1 << CHUNK_BITS;
const CHUNK_MASK: u32 = (CHUNK_LEN - 1) as u32;

/// Chunked arena of search nodes.
///
/// Slots are `Option<Node>` so that a node can be *moved* out during
/// migration rather than cloned -- re-rooting keeps a subtree that can run to
/// millions of nodes and cloning a gamestate each would cost more than the
/// search. `Node`'s move list is a `Vec`, whose non-null pointer gives
/// `Option` a niche, so the slot is the same size as the node itself;
/// `option_is_free` asserts that.
pub struct Arena<G, M> {
    chunks: Vec<Vec<Option<Node<G, M>>>>,
    len: usize,
}

impl<G, M> Default for Arena<G, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G, M> Arena<G, M> {
    pub fn new() -> Self {
        Arena {
            chunks: Vec::new(),
            len: 0,
        }
    }

    /// A new arena holding a single node, and its id.
    pub fn with_root(root: Node<G, M>) -> (Self, NodeId) {
        let mut arena = Self::new();
        let id = arena.alloc(root);
        (arena, id)
    }

    /// Number of live nodes. Slots emptied by [`Arena::take`] still count, so
    /// this is only meaningful outside a migration.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn alloc(&mut self, node: Node<G, M>) -> NodeId {
        if self.chunks.last().is_none_or(|c| c.len() == CHUNK_LEN) {
            self.chunks.push(Vec::with_capacity(CHUNK_LEN));
        }
        let chunk = self.chunks.len() - 1;
        let slot = self.chunks[chunk].len();
        let id = (chunk << CHUNK_BITS) | slot;
        assert!(id <= NodeId::MAX as usize, "arena exceeded 2^32 nodes");
        self.chunks[chunk].push(Some(node));
        self.len += 1;
        id as NodeId
    }

    #[inline]
    pub fn get(&self, id: NodeId) -> &Node<G, M> {
        self.chunks[(id >> CHUNK_BITS) as usize][(id & CHUNK_MASK) as usize]
            .as_ref()
            .expect("node was migrated out of this arena")
    }

    #[inline]
    pub fn get_mut(&mut self, id: NodeId) -> &mut Node<G, M> {
        self.chunks[(id >> CHUNK_BITS) as usize][(id & CHUNK_MASK) as usize]
            .as_mut()
            .expect("node was migrated out of this arena")
    }

    /// Move a node out, leaving the slot empty. Only for migration: any other
    /// id still pointing at this node is now dangling in the logical sense and
    /// will panic on access rather than read stale data.
    fn take(&mut self, id: NodeId) -> Node<G, M> {
        let node = self.chunks[(id >> CHUNK_BITS) as usize][(id & CHUNK_MASK) as usize]
            .take()
            .expect("node was migrated out of this arena");
        self.len -= 1;
        node
    }
}

/// Move the subtree rooted at `id` out of `src` and into `dst`, returning its
/// new root id.
///
/// Nodes are moved, not copied: the gamestate, the move list and the children
/// vector all transfer without reallocating. What is left behind in `src` is
/// exactly the discarded part of the tree, which the caller frees by dropping
/// `src` whole.
pub fn migrate<G, M>(src: &mut Arena<G, M>, dst: &mut Arena<G, M>, id: NodeId) -> NodeId {
    let mut node = src.take(id);
    // Taken out first so the recursion is not holding a borrow of either arena.
    let children = std::mem::take(&mut node.children);
    let new_id = dst.alloc(node);
    let mut moved = Vec::with_capacity(children.len());
    for (m, child) in children {
        let child = migrate(src, dst, child);
        moved.push((m, child));
    }
    dst.get_mut(new_id).children = moved;
    new_id
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::games::mancala::{Mancala, MancalaMove};

    /// The `Option` around each slot is what makes migration a move rather
    /// than a clone. It is only affordable because `Vec`'s non-null pointer
    /// gives it a niche -- if that ever stops holding, every node in the tree
    /// silently grows.
    #[test]
    fn option_is_free() {
        assert_eq!(
            std::mem::size_of::<Option<Node<Mancala, MancalaMove>>>(),
            std::mem::size_of::<Node<Mancala, MancalaMove>>(),
        );
    }

    #[test]
    fn ids_survive_chunk_boundaries() {
        let mut arena = Arena::<Mancala, MancalaMove>::new();
        let ids: Vec<NodeId> = (0..CHUNK_LEN * 3 + 7)
            .map(|i| {
                let mut node = Node::new(Mancala::new());
                node.descendants = i as u32;
                arena.alloc(node)
            })
            .collect();
        assert_eq!(arena.len(), ids.len());
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(arena.get(*id).descendants, i as u32);
        }
    }

    #[test]
    fn migration_keeps_the_subtree_and_drops_the_rest() {
        let mut arena = Arena::<Mancala, MancalaMove>::new();
        let root = arena.alloc(Node::new(Mancala::new()));
        let mut kept_ids = Vec::new();
        for i in 0..4u32 {
            let mut child = Node::new(Mancala::new());
            child.descendants = i;
            let id = arena.alloc(child);
            let mut grandchild = Node::new(Mancala::new());
            grandchild.descendants = 100 + i;
            let gid = arena.alloc(grandchild);
            arena.get_mut(id).children.push((MancalaMove(0), gid));
            arena.get_mut(root).children.push((MancalaMove(0), id));
            kept_ids.push(id);
        }
        assert_eq!(arena.len(), 9);

        let keep = kept_ids[2];
        let mut dst = Arena::new();
        let new_root = migrate(&mut arena, &mut dst, keep);
        drop(arena);

        assert_eq!(dst.len(), 2);
        assert_eq!(dst.get(new_root).descendants, 2);
        let child = dst.get(new_root).children[0].1;
        assert_eq!(dst.get(child).descendants, 102);
    }
}
