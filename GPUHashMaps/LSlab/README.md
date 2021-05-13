# LSlab

Changes from Slab:
- Added ilock integer to slab struct and added padding to slab to accommodate
- Lock from first lane and then sync with warp
- Unlock after done synchronizing with warp at end of round