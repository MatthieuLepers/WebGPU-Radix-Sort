(function(c,l){typeof exports=="object"&&typeof module<"u"?l(exports):typeof define=="function"&&define.amd?define(["exports"],l):(c=typeof globalThis<"u"?globalThis:c||self,l(c.RadixSort={}))})(this,function(c){"use strict";function l(n,e){const i={x:e,y:1};if(e>n.limits.maxComputeWorkgroupsPerDimension){const r=Math.floor(Math.sqrt(e)),t=Math.ceil(e/r);i.x=r,i.y=t}return i}function _({device:n,label:e,data:i,usage:r=0}){const t=n.createBuffer({label:e,usage:r,size:i.length*4,mappedAtCreation:!0});return new Uint32Array(t.getMappedRange()).set(i),t.unmap(),t}const b=`
@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

var<workgroup> temp: array<u32, 2u * ITEMS_PER_WORKGROUP>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  let ELM_TID = TID * 2u; // Element pair local ID
  let ELM_GID = GID * 2u; // Element pair global ID

  // Load input to shared memory
  temp[ELM_TID]      = select(items[ELM_GID], 0u, ELM_GID >= ELEMENT_COUNT);
  temp[ELM_TID + 1u] = select(items[ELM_GID + 1u], 0u, ELM_GID + 1u >= ELEMENT_COUNT);

  var offset: u32 = 1u;

  // Up-sweep (reduce) phase
  for (var d: u32 = ITEMS_PER_WORKGROUP >> 1u; d > 0u; d >>= 1u) {
    workgroupBarrier();

    if (TID < d) {
      var ai: u32 = offset * (ELM_TID + 1u) - 1u;
      var bi: u32 = offset * (ELM_TID + 2u) - 1u;
      temp[bi] += temp[ai];
    }

    offset *= 2u;
  }

    // Save workgroup sum and clear last element
  if (TID == 0u) {
    let last_offset = ITEMS_PER_WORKGROUP - 1u;

    blockSums[WORKGROUP_ID] = temp[last_offset];
    temp[last_offset] = 0u;
  }

  // Down-sweep phase
  for (var d: u32 = 1u; d < ITEMS_PER_WORKGROUP; d *= 2u) {
    offset >>= 1u;
    workgroupBarrier();

    if (TID < d) {
      var ai: u32 = offset * (ELM_TID + 1u) - 1u;
      var bi: u32 = offset * (ELM_TID + 2u) - 1u;

      let t: u32 = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  workgroupBarrier();

  // Copy result from shared memory to global memory
  if (ELM_GID >= ELEMENT_COUNT) {
    return;
  }
  items[ELM_GID] = temp[ELM_TID];

  if (ELM_GID + 1u >= ELEMENT_COUNT) {
    return;
  }
  items[ELM_GID + 1u] = temp[ELM_TID + 1u];
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  let ELM_ID = GID * 2u;

  if (ELM_ID >= ELEMENT_COUNT) {
    return;
  }

  let blockSum = blockSums[WORKGROUP_ID];

  items[ELM_ID] += blockSum;

  if (ELM_ID + 1u >= ELEMENT_COUNT) {
    return;
  }

  items[ELM_ID + 1u] += blockSum;
}
`,g=`
@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

const NUM_BANKS: u32 = 32u;
const LOG_NUM_BANKS: u32 = 5u;

fn get_offset(offset: u32) -> u32 {
  // return offset >> LOG_NUM_BANKS; // Conflict-free
  return (offset >> NUM_BANKS) + (offset >> (2u * LOG_NUM_BANKS)); // Zero bank conflict
}

var<workgroup> temp: array<u32, 2u * ITEMS_PER_WORKGROUP>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  let ELM_TID = TID * 2u; // Element pair local ID
  let ELM_GID = GID * 2u; // Element pair global ID

  // Load input to shared memory
  let ai: u32 = TID;
  let bi: u32 = TID + (ITEMS_PER_WORKGROUP >> 1u);
  let s_ai = ai + get_offset(ai);
  let s_bi = bi + get_offset(bi);
  let g_ai = ai + WID * 2u;
  let g_bi = bi + WID * 2u;
  temp[s_ai] = select(items[g_ai], 0u, g_ai >= ELEMENT_COUNT);
  temp[s_bi] = select(items[g_bi], 0u, g_bi >= ELEMENT_COUNT);

  var offset: u32 = 1u;

  // Up-sweep (reduce) phase
  for (var d: u32 = ITEMS_PER_WORKGROUP >> 1u; d > 0u; d >>= 1u) {
    workgroupBarrier();

    if (TID < d) {
      var ai: u32 = offset * (ELM_TID + 1u) - 1u;
      var bi: u32 = offset * (ELM_TID + 2u) - 1u;
      ai += get_offset(ai);
      bi += get_offset(bi);
      temp[bi] += temp[ai];
    }

    offset *= 2u;
  }

  // Save workgroup sum and clear last element
  if (TID == 0u) {
    var last_offset = ITEMS_PER_WORKGROUP - 1u;
    last_offset += get_offset(last_offset);

    blockSums[WORKGROUP_ID] = temp[last_offset];
    temp[last_offset] = 0u;
  }

  // Down-sweep phase
  for (var d: u32 = 1u; d < ITEMS_PER_WORKGROUP; d *= 2u) {
    offset >>= 1u;
    workgroupBarrier();

    if (TID < d) {
      var ai: u32 = offset * (ELM_TID + 1u) - 1u;
      var bi: u32 = offset * (ELM_TID + 2u) - 1u;
      ai += get_offset(ai);
      bi += get_offset(bi);

      let t: u32 = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  workgroupBarrier();

  // Copy result from shared memory to global memory
  if (g_ai < ELEMENT_COUNT) {
    items[g_ai] = temp[s_ai];
  }
  if (g_bi < ELEMENT_COUNT) {
    items[g_bi] = temp[s_bi];
  }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
  let GID = WID + TID; // Global thread ID

  let ELM_ID = GID * 2u;

  if (ELM_ID >= ELEMENT_COUNT) {
    return;
  }

  let blockSum = blockSums[WORKGROUP_ID];

  items[ELM_ID] += blockSum;

  if (ELM_ID + 1u >= ELEMENT_COUNT) {
    return;
  }

  items[ELM_ID + 1u] += blockSum;
}
`;class h{device;count;workgroupSize;pipelines=[];shaderModules={};constructor({device:e,count:i,workgroupSize:r={x:16,y:16}}){this.device=e,this.count=i,this.workgroupSize=r}get workgroupCount(){return Math.ceil(this.count/this.threadsPerWorkgroup)}get threadsPerWorkgroup(){return this.workgroupSize.x*this.workgroupSize.y}get itemsPerWorkgroup(){return 2*this.threadsPerWorkgroup}}class O extends h{constructor({device:e,count:i,workgroupSize:r={x:16,y:16},data:t,avoidBankConflicts:s=!1}){if(super({device:e,count:i,workgroupSize:r}),Math.log2(this.threadsPerWorkgroup)%1!==0)throw new Error(`workgroupSize.x * workgroupSize.y must be a power of two. (current: ${this.threadsPerWorkgroup})`);this.shaderModules.prefixSum=this.device.createShaderModule({label:"prefix-sum",code:s?g:b}),this.createPassRecursive(t,i)}createPassRecursive(e,i){const r=Math.ceil(i/this.itemsPerWorkgroup),t=l(this.device,r),s=this.device.createBuffer({label:"prefix-sum-block-sum",size:r*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),o=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),u=this.device.createBindGroup({label:"prefix-sum-bind-group",layout:o,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:s}}]}),a=this.device.createPipelineLayout({bindGroupLayouts:[o]}),f=this.device.createComputePipeline({label:"prefix-sum-scan-pipeline",layout:a,compute:{module:this.shaderModules.prefixSum,entryPoint:"reduce_downsweep",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ITEMS_PER_WORKGROUP:this.itemsPerWorkgroup,ELEMENT_COUNT:i}}});if(this.pipelines.push({pipeline:f,bindGroup:u,dispatchSize:t}),r>1){this.createPassRecursive(s,r);const p=this.device.createComputePipeline({label:"prefix-sum-add-block-pipeline",layout:a,compute:{module:this.shaderModules.prefixSum,entryPoint:"add_block_sums",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:i}}});this.pipelines.push({pipeline:p,bindGroup:u,dispatchSize:t})}}getDispatchChain(){return this.pipelines.flatMap(e=>[e.dispatchSize.x,e.dispatchSize.y,1])}dispatch(e,i,r=0){this.pipelines.forEach(({pipeline:t,bindGroup:s,dispatchSize:o},u)=>{e.setPipeline(t),e.setBindGroup(0,s),i?e.dispatchWorkgroupsIndirect(i,r+u*3*4):e.dispatchWorkgroups(o.x,o.y,1)})}}const S=(n=!1,e=!1,i="full")=>`
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> original: array<u32>;
@group(0) @binding(3) var<storage, read_write> is_sorted: u32;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;
override START_ELEMENT: u32;

var<workgroup> s_data: array<u32, THREADS_PER_WORKGROUP>;

// Reset dispatch buffer and is_sorted flag
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reset(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  if (TID >= ELEMENT_COUNT) {
    return;
  }

  if (TID == 0) {
    is_sorted = 0u;
  }

  let ELM_ID = TID * 3;

  output[ELM_ID] = original[ELM_ID];
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn check_sort(
  @builtin(workgroup_id) w_id: vec3<u32>,
  @builtin(num_workgroups) w_dim: vec3<u32>,
  @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
  let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
  let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP + START_ELEMENT;
  let GID = TID + WID; // Global thread ID

  // Load data into shared memory
  ${n?m:"s_data[TID] = select(0u, input[GID], GID < ELEMENT_COUNT);"}

  // Perform parallel reduction
  for (var d = 1u; d < THREADS_PER_WORKGROUP; d *= 2u) {
    workgroupBarrier();  
    if (TID % (2u * d) == 0u) {
      s_data[TID] += s_data[TID + d];
    }
  }
  workgroupBarrier();

  // Write reduction result
  ${e?P(i):E}
}`,E=`
  if (TID == 0) {
    output[WORKGROUP_ID] = s_data[0];
  }
`,m=`
  let LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;

  // Load current element into shared memory
  // Also load next element for comparison
  let elm = select(0u, input[GID], GID < ELEMENT_COUNT);
  let next = select(0u, input[GID + 1], GID < ELEMENT_COUNT-1);
  s_data[TID] = elm;
  workgroupBarrier();

  s_data[TID] = select(0u, 1u, GID < ELEMENT_COUNT-1 && elm > next);
`,P=n=>`
  let fullDispatchLength = arrayLength(&output);
  let dispatchIndex = TID * 3;

  if (dispatchIndex >= fullDispatchLength) {
    return;
  }

  ${n=="full"?T:I}
`,I=`
  output[dispatchIndex] = select(0, original[dispatchIndex], s_data[0] == 0 && is_sorted == 0u);
`,T=`
  if (TID == 0 && s_data[0] == 0) {
    is_sorted = 1u;
  }

  output[dispatchIndex] = select(0, original[dispatchIndex], s_data[0] != 0);
`;class d extends h{start;mode;buffers={};outputs=[];constructor({device:e,count:i,workgroupSize:r={x:16,y:16},data:t,result:s,original:o,isSorted:u,start:a=0,mode:f="full"}){super({device:e,count:i,workgroupSize:r}),this.start=a,this.mode=f,this.buffers={data:t,result:s,original:o,isSorted:u},this.createPassesRecursive(t,i)}static findOptimalDispatchChain(e,i,r){const t=r.x*r.y,s=[];do{const o=Math.ceil(i/t),u=l(e,o);s.push(u.x,u.y,1),i=o}while(i>1);return s}createPassesRecursive(e,i,r=0){const t=Math.ceil(i/this.threadsPerWorkgroup),s=r===0,o=t<=1,u=`check-sort-${this.mode}-${r}`,a=o?this.buffers.result:this.device.createBuffer({label:u,size:t*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),f=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},...o?[{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),p=this.device.createBindGroup({layout:f,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:a}},...o?[{binding:2,resource:{buffer:this.buffers.original}},{binding:3,resource:{buffer:this.buffers.isSorted}}]:[]]}),k=this.device.createPipelineLayout({bindGroupLayouts:[f]}),v=s?this.start+i:i,w=s?this.start:0,x=this.device.createComputePipeline({layout:k,compute:{module:this.device.createShaderModule({label:u,code:S(s,o,this.mode)}),entryPoint:this.mode=="reset"?"reset":"check_sort",constants:{ELEMENT_COUNT:v,WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,...this.mode!=="reset"&&{THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,START_ELEMENT:w}}}});this.outputs.push(a),this.pipelines.push({pipeline:x,bindGroup:p}),o||this.createPassesRecursive(a,t,r+1)}dispatch(e,i,r=0){this.pipelines.forEach(({pipeline:t,bindGroup:s},o)=>{const u=this.mode!=="reset"&&(this.mode==="full"||o<this.pipelines.length-1);e.setPipeline(t),e.setBindGroup(0,s),u&&i?e.dispatchWorkgroupsIndirect(i,r+o*3*4):e.dispatchWorkgroups(1,1,1)})}}const U=`
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

var<workgroup> s_prefix_sum: array<u32, 2u * (THREADS_PER_WORKGROUP + 1u)>;

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
  let elm = select(input[GID], 0u, GID >= ELEMENT_COUNT);
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
    local_prefix_sums[GID] = bit_prefix_sums[extract_bits];
  }
}
`,D=`
@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> local_prefix_sums: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<storage, read_write> values: array<u32>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

var<workgroup> s_prefix_sum: array<u32, 2u * (THREADS_PER_WORKGROUP + 1u)>;
var<workgroup> s_prefix_sum_scan: array<u32, 4>;

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
  var elm: u32 = 0u;
  var val: u32 = 0u;
  if (GID < ELEMENT_COUNT) {
    elm = input[GID];
    val = values[GID];
  }
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

  let prefix_sum = bit_prefix_sums[extract_bits];   

  // Scan bit prefix sums
  if (TID == LAST_THREAD) {
    var sum: u32 = 0u;
    bit_prefix_sums[extract_bits] += 1u;

    for (var i: u32 = 0u; i < 4u; i++) {
      s_prefix_sum_scan[i] = sum;
      sum += bit_prefix_sums[i];
    }
  }
  workgroupBarrier();

  if (GID < ELEMENT_COUNT) {
    // Compute new position
    let new_pos: u32 = prefix_sum + s_prefix_sum_scan[extract_bits];

    // Shuffle elements locally
    input[WID + new_pos] = elm;
    values[WID + new_pos] = val;
    local_prefix_sums[WID + new_pos] = prefix_sum;
  }
}
`,R=`
@group(0) @binding(0) var<storage, read> inputKeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(2) var<storage, read> local_prefix_sum: array<u32>;
@group(0) @binding(3) var<storage, read> prefix_block_sum: array<u32>;
@group(0) @binding(4) var<storage, read> inputValues: array<u32>;
@group(0) @binding(5) var<storage, read_write> outputValues: array<u32>;

override WORKGROUP_COUNT: u32;
override THREADS_PER_WORKGROUP: u32;
override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override CURRENT_BIT: u32;
override ELEMENT_COUNT: u32;

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

  let k = inputKeys[GID];
  let v = inputValues[GID];

  let local_prefix = local_prefix_sum[GID];

  // Calculate new position
  let extract_bits = (k >> CURRENT_BIT) & 0x3u;
  let pid = extract_bits * WORKGROUP_COUNT + WORKGROUP_ID;
  let sorted_position = prefix_block_sum[pid] + local_prefix;

  outputKeys[sorted_position] = k;
  outputValues[sorted_position] = v;
}
`;class G extends h{bitCount;checkOrder=!1;localShuffle=!1;avoidBankConflicts=!1;prefixBlockWorkgroupCount;hasValues;dispatchSize={x:1,y:1};dispatchOffsets={radixSort:0,checkSortFast:3*4,prefixSum:6*4};initialDispatch=[];kernels={};buffers={};texture;constructor({device:e,count:i,workgroupSize:r={x:16,y:16},texture:t,keys:s,values:o,bitCount:u=32,checkOrder:a=!1,localShuffle:f=!1,avoidBankConflicts:p=!1}){if(super({device:e,count:i,workgroupSize:r}),!e)throw new Error("No device provided");if(!s&&!t)throw new Error("No keys buffer or texture provided");if(!Number.isInteger(i)||i<=0)throw new Error("Invalid count parameter");if(!Number.isInteger(u)||u<=0||u>32)throw new Error(`Invalid bitCount parameter: ${u}`);if(!Number.isInteger(r.x)||!Number.isInteger(r.y))throw new Error("Invalid workgroupSize parameter");if(u%4!=0)throw new Error("bitCount must be a multiple of 4");this.bitCount=u,this.checkOrder=a,this.localShuffle=f,this.avoidBankConflicts=p,this.prefixBlockWorkgroupCount=4*this.workgroupCount,this.hasValues=!!o||!!t,this.texture=t,this.buffers={keys:s,values:o},this.createShaderModules(),this.createPipelines()}createShaderModules(){const e=r=>r.split(`
`).filter(t=>!t.toLowerCase().includes("values")).join(`
`),i=this.localShuffle?D:U;this.shaderModules={blockSum:this.device.createShaderModule({label:"radix-sort-block-sum",code:this.hasValues?i:e(i)}),reorder:this.device.createShaderModule({label:"radix-sort-reorder",code:this.hasValues?R:e(R)})}}createPipelines(){this.createPrefixSumKernel();const e=this.calculateDispatchSizes();this.createBuffers(e),this.createCheckSortKernels(e);for(let i=0;i<this.bitCount;i+=2){const r=i%4==0,t=r?this.buffers.keys:this.buffers.tmpKeys,s=r?this.buffers.values:this.buffers.tmpValues,o=r?this.buffers.tmpKeys:this.buffers.keys,u=r?this.buffers.tmpValues:this.buffers.values,a=this.createBlockSumPipeline(t,s,i),f=this.createReorderPipeline(t,s,o,u,i);this.pipelines.push(a,f)}}createPrefixSumKernel(){const e=this.device.createBuffer({label:"radix-sort-prefix-block-sum",size:this.prefixBlockWorkgroupCount*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),i=new O({device:this.device,data:e,count:this.prefixBlockWorkgroupCount,workgroupSize:this.workgroupSize,avoidBankConflicts:this.avoidBankConflicts});this.kernels.prefixSum=i,this.buffers.prefixBlockSum=e}calculateDispatchSizes(){const e=l(this.device,this.workgroupCount),i=this.kernels.prefixSum.getDispatchChain(),r=Math.min(this.count,this.threadsPerWorkgroup*4),t=this.count-r,s=r-1,o=d.findOptimalDispatchChain(this.device,r,this.workgroupSize),u=d.findOptimalDispatchChain(this.device,t,this.workgroupSize),a=[e.x,e.y,1,...o.slice(0,3),...i];return this.dispatchOffsets={radixSort:0,checkSortFast:3*4,prefixSum:6*4},this.dispatchSize=e,this.initialDispatch=a,{initialDispatch:a,dispatchSizesFull:u,checkSortFastCount:r,checkSortFullCount:t,startFull:s}}createBuffers(e){const i=this.device.createBuffer({label:"radix-sort-tmp-keys",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),r=this.hasValues?this.device.createBuffer({label:"radix-sort-tmp-values",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}):void 0,t=this.device.createBuffer({label:"radix-sort-local-prefix-sum",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});if(this.buffers.tmpKeys=i,this.buffers.tmpValues=r,this.buffers.localPrefixSum=t,!this.checkOrder)return;const s=_({device:this.device,label:"radix-sort-dispatch-size",data:e.initialDispatch,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.INDIRECT}),o=_({device:this.device,label:"radix-sort-dispatch-size-original",data:e.initialDispatch,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),u=_({label:"check-sort-full-dispatch-size",device:this.device,data:e.dispatchSizesFull,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.INDIRECT}),a=_({label:"check-sort-full-dispatch-size-original",device:this.device,data:e.dispatchSizesFull,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),f=_({label:"is-sorted",device:this.device,data:[0],usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});this.buffers.dispatchSize=s,this.buffers.originalDispatchSize=o,this.buffers.checkSortFullDispatchSize=u,this.buffers.originalCheckSortFullDispatchSize=a,this.buffers.isSorted=f}createCheckSortKernels(e){if(!this.checkOrder)return;const{checkSortFastCount:i,checkSortFullCount:r,startFull:t}=e,s=new d({mode:"full",device:this.device,data:this.buffers.keys,result:this.buffers.dispatchSize,original:this.buffers.originalDispatchSize,isSorted:this.buffers.isSorted,count:r,start:t,workgroupSize:this.workgroupSize}),o=new d({mode:"fast",device:this.device,data:this.buffers.keys,result:this.buffers.checkSortFullDispatchSize,original:this.buffers.originalCheckSortFullDispatchSize,isSorted:this.buffers.isSorted,count:i,workgroupSize:this.workgroupSize}),u=this.initialDispatch.length/3;if(o.threadsPerWorkgroup<s.pipelines.length||s.threadsPerWorkgroup<u){console.warn("Warning: workgroup size is too small to enable check sort optimization, disabling..."),this.checkOrder=!1;return}const a=new d({mode:"reset",device:this.device,data:this.buffers.keys,original:this.buffers.originalDispatchSize,result:this.buffers.dispatchSize,isSorted:this.buffers.isSorted,count:u,workgroupSize:l(this.device,u)});this.kernels.checkSort={reset:a,fast:o,full:s}}createBlockSumPipeline(e,i,r){const t=this.device.createBindGroupLayout({label:"radix-sort-block-sum",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:this.localShuffle?"storage":"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},...this.localShuffle&&this.hasValues?[{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),s=this.device.createBindGroup({layout:t,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.buffers.localPrefixSum}},{binding:2,resource:{buffer:this.buffers.prefixBlockSum}},...this.localShuffle&&this.hasValues?[{binding:3,resource:{buffer:i}}]:[]]}),o=this.device.createPipelineLayout({bindGroupLayouts:[t]});return{pipeline:this.device.createComputePipeline({label:"radix-sort-block-sum",layout:o,compute:{module:this.shaderModules.blockSum,entryPoint:"radix_sort",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,WORKGROUP_COUNT:this.workgroupCount,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:this.count,CURRENT_BIT:r}}}),bindGroup:s}}createReorderPipeline(e,i,r,t,s){const o=this.device.createBindGroupLayout({label:"radix-sort-reorder",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},...this.hasValues?[{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),u=this.device.createBindGroup({layout:o,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:r}},{binding:2,resource:{buffer:this.buffers.localPrefixSum}},{binding:3,resource:{buffer:this.buffers.prefixBlockSum}},...this.hasValues?[{binding:4,resource:{buffer:i}},{binding:5,resource:{buffer:t}}]:[]]}),a=this.device.createPipelineLayout({bindGroupLayouts:[o]});return{pipeline:this.device.createComputePipeline({label:"radix-sort-reorder",layout:a,compute:{module:this.shaderModules.reorder,entryPoint:"radix_sort_reorder",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,WORKGROUP_COUNT:this.workgroupCount,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:this.count,CURRENT_BIT:s}}}),bindGroup:u}}dispatch(e){this.checkOrder?this.#i(e):this.#e(e)}#e(e){for(let i=0;i<this.bitCount/2;i+=1){const r=this.pipelines[i*2],t=this.pipelines[i*2+1];e.setPipeline(r.pipeline),e.setBindGroup(0,r.bindGroup),e.dispatchWorkgroups(this.dispatchSize.x,this.dispatchSize.y,1),this.kernels.prefixSum.dispatch(e),e.setPipeline(t.pipeline),e.setBindGroup(0,t.bindGroup),e.dispatchWorkgroups(this.dispatchSize.x,this.dispatchSize.y,1)}}#i(e){this.kernels.checkSort.reset.dispatch(e);for(let i=0;i<this.bitCount/2;i++){const r=this.pipelines[i*2],t=this.pipelines[i*2+1];i%2==0&&(this.kernels.checkSort.fast.dispatch(e,this.buffers.dispatchSize,this.dispatchOffsets.checkSortFast),this.kernels.checkSort.full.dispatch(e,this.buffers.checkSortFullDispatchSize)),e.setPipeline(r.pipeline),e.setBindGroup(0,r.bindGroup),e.dispatchWorkgroupsIndirect(this.buffers.dispatchSize,this.dispatchOffsets.radixSort),this.kernels.prefixSum.dispatch(e,this.buffers.dispatchSize,this.dispatchOffsets.prefixSum),e.setPipeline(t.pipeline),e.setBindGroup(0,t.bindGroup),e.dispatchWorkgroupsIndirect(this.buffers.dispatchSize,this.dispatchOffsets.radixSort)}}}c.PrefixSumKernel=O,c.RadixSortKernel=G,Object.defineProperty(c,Symbol.toStringTag,{value:"Module"})});
//# sourceMappingURL=radix-sort.umd.js.map
