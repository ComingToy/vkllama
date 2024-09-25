#define u8BufToU32(buf, idx)                                                  \
  ((uint (buf[idx + 3]) << 24) | (uint (buf[idx + 2]) << 16)                  \
   | (uint (buf[idx + 1]) << 8) | (uint (buf[idx])))

#define Q8_0_ITEMS_PER_BLOCK 32
#define Q8_0_BYTES_PER_BLOCK 36
#define Q8_0_SCALE_BYTES 4
