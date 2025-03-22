const radixSortSource = (dataType: 'buffer' | 'texture') => /* wgsl */ `
${
  dataType === 'buffer'
    ? `
      @group(0) @binding(0) var<storage, read> input: array<u32>;
      @group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;
    `
    : `
      @group(0) @binding(0) var input: texture_storage_2d<rg32uint, read>;
      @group(0) @binding(1) var local_prefix_sums: texture_storage_2d<r32uint, write>;
    `
}
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

var<workgroup> s_prefix_sum: array<u32, 2u * (THREADS_PER_WORKGROUP + 1u)>;

fn getInput(index: u32) -> u32 {
  ${
    dataType === 'buffer'
      ? `return input[index];`
      : `
        let dimX = textureDimensions(input).r;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        return textureLoad(input, vec2<i32>(x, y)).x;
      `
  }
}

fn setLocalPrefixSum(index: u32, val: u32) {
  ${
    dataType === 'buffer'
      ? 'local_prefix_sums[index] = val;'
      : `
        let dimX = textureDimensions(local_prefix_sums).x;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        textureStore(local_prefix_sums, vec2<i32>(x, y), vec4<u32>(val, 0u, 0u, 0u));
      `
  }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  // Extract 2 bits from the input
  let elm = select(getInput(GID), 0u, GID >= ELEMENT_COUNT);
  let extract_bits: u32 = (elm >> CURRENT_BIT) & 0x3u;

  var bit_prefix_sums = array<u32, 4>(0, 0, 0, 0);

  // If the workgroup is inactive, prevent block_sums buffer update
  var LAST_THREAD: u32 = 0xffffffffu;

  if (WORKGROUP_ID < WORKGROUP_COUNT) {
    // Otherwise store the index of the last active thread in the workgroup
    LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1u;
  }

  // Initialize parameters for double-buffering
  let TPW = THREADS_PER_WORKGROUP + 1u;
  var swapOffset: u32 = 0u;
  var inOffset:  u32 = TID;
  var outOffset: u32 = TID + TPW;

  // 4-way prefix sum
  for (var b: u32 = 0u; b < 4u; b++) {
    // Initialize local prefix with bitmask
    let bitmask = select(0u, 1u, extract_bits == b);
    s_prefix_sum[inOffset + 1u] = bitmask;
    workgroupBarrier();

    var prefix_sum: u32 = 0u;

    // Prefix sum
    for (var offset: u32 = 1u; offset < THREADS_PER_WORKGROUP; offset *= 2u) {
      if (TID >= offset) {
        prefix_sum = s_prefix_sum[inOffset] + s_prefix_sum[inOffset - offset];
      } else {
        prefix_sum = s_prefix_sum[inOffset];
      }

      s_prefix_sum[outOffset] = prefix_sum;

      // Swap buffers
      outOffset = inOffset;
      swapOffset = TPW - swapOffset;
      inOffset = TID + swapOffset;

      workgroupBarrier();
    }

    // Store prefix sum for current bit
    bit_prefix_sums[b] = prefix_sum;

    if (TID == LAST_THREAD) {
      // Store block sum to global memory
      let total_sum: u32 = prefix_sum + bitmask;
      block_sums[b * WORKGROUP_COUNT + WORKGROUP_ID] = total_sum;
    }

    // Swap buffers
    outOffset = inOffset;
    swapOffset = TPW - swapOffset;
    inOffset = TID + swapOffset;
  }

  if (GID < ELEMENT_COUNT) {
    // Store local prefix sum to global memory
    setLocalPrefixSum(GID, bit_prefix_sums[extract_bits]);
  }
}
`;

export default radixSortSource;
