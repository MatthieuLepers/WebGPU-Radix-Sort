const reorderSource = (dataType: 'buffer' | 'texture') => `
${
  dataType === 'buffer'
    ? `
      @group(0) @binding(0) var<storage, read> inputKeys: array<u32>;
      @group(0) @binding(1) var<storage, read_write> outputKeys: array<u32>;
      @group(0) @binding(2) var<storage, read> local_prefix_sum: array<u32>;
    `
    : `
      @group(0) @binding(0) var input: texture_storage_2d<rg32uint, read>;
      @group(0) @binding(1) var output: texture_storage_2d<rg32uint, write>;
      @group(0) @binding(2) var local_prefix_sum: texture_storage_2d<r32uint, read_write>;
    `
}
@group(0) @binding(3) var<storage, read> prefix_block_sum: array<u32>;
${
  dataType === 'buffer'
    ? `
      @group(0) @binding(4) var<storage, read> inputValues: array<u32>;
      @group(0) @binding(5) var<storage, read_write> outputValues: array<u32>;
    `
    : ''
}

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

fn getInput(index: u32) -> vec2<u32> {
  ${
    dataType === 'buffer'
      ? `
        let result: vec2<u32> = vec2<u32>(
          inputKeys[index],
          inputValues[index]
        );
        return result;
      `
      : `
        let dimX = textureDimensions(input).x;
        let x = i32(index % dimX);
        let y = i32(index / dimY);
        return textureLoad(input, vec2<i32>(x, y)).xy;
      `
  }
}

fn setOutput(index: u32, key: u32, val: u32) {
  ${
    dataType === 'buffer'
      ? `
        outputKeys[index] = key;
        outputValues[index] = val;
      `
      : `
        let dimX = textureDimensions(output).x;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        textureStore(output, vec2<i32>(x, y), vec4<u32>(key, val, 0u, 0u));
      `
  }
}

fn getLocalPrefixSum(index: u32) -> u32 {
  ${
    dataType === 'buffer'
      ? 'return local_prefix_sum[index];'
      : `
        let dimX = textureDimensions(local_prefix_sum).x;
        let x = i32(index % dimX);
        let y = i32(index / dimX);
        return textureLoad(local_prefix_sum, vec2<i32>(x, y)).x;
      `
  }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn radix_sort_reorder(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  if (GID >= ELEMENT_COUNT) { return; }

  let input = getInput(GID);

  let local_prefix = getLocalPrefixSum(GID);

  // Calculate new position
  let extract_bits = (input.x >> CURRENT_BIT) & 0x3u;
  let pid = extract_bits * WORKGROUP_COUNT + WORKGROUP_ID;
  let sorted_position = prefix_block_sum[pid] + local_prefix;

  setOutput(sorted_position, input.x, input.y);
}
`;

export default reorderSource;
