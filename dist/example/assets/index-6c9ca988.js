var L=Object.defineProperty;var K=(a,r,e)=>r in a?L(a,r,{enumerable:!0,configurable:!0,writable:!0,value:e}):a[r]=e;var f=(a,r,e)=>(K(a,typeof r!="symbol"?r+"":r,e),e),B=(a,r,e)=>{if(!r.has(a))throw TypeError("Cannot "+e)};var k=(a,r,e)=>{if(r.has(a))throw TypeError("Cannot add the same private member more than once");r instanceof WeakSet?r.add(a):r.set(a,e)};var v=(a,r,e)=>(B(a,r,"access private method"),e);(function(){const r=document.createElement("link").relList;if(r&&r.supports&&r.supports("modulepreload"))return;for(const t of document.querySelectorAll('link[rel="modulepreload"]'))i(t);new MutationObserver(t=>{for(const s of t)if(s.type==="childList")for(const u of s.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&i(u)}).observe(document,{childList:!0,subtree:!0});function e(t){const s={};return t.integrity&&(s.integrity=t.integrity),t.referrerPolicy&&(s.referrerPolicy=t.referrerPolicy),t.crossOrigin==="use-credentials"?s.credentials="include":t.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function i(t){if(t.ep)return;t.ep=!0;const s=e(t);fetch(t.href,s)}})();function P(a,r){const e={x:r,y:1};if(r>a.limits.maxComputeWorkgroupsPerDimension){const i=Math.floor(Math.sqrt(r)),t=Math.ceil(r/i);e.x=i,e.y=t}return e}function S({device:a,label:r,data:e,usage:i=0}){const t=a.createBuffer({label:r,usage:i,size:e.length*4,mappedAtCreation:!0});return new Uint32Array(t.getMappedRange()).set(e),t.unmap(),t}const N=`
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
`,z=`
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
`;class G{constructor({device:r,count:e,workgroupSize:i={x:16,y:16}}){f(this,"device");f(this,"count");f(this,"workgroupSize");f(this,"pipelines",[]);f(this,"shaderModules",{});this.device=r,this.count=e,this.workgroupSize=i}get workgroupCount(){return Math.ceil(this.count/this.threadsPerWorkgroup)}get threadsPerWorkgroup(){return this.workgroupSize.x*this.workgroupSize.y}get itemsPerWorkgroup(){return 2*this.threadsPerWorkgroup}}class A extends G{constructor({device:r,count:e,workgroupSize:i={x:16,y:16},data:t,avoidBankConflicts:s=!1}){if(super({device:r,count:e,workgroupSize:i}),Math.log2(this.threadsPerWorkgroup)%1!==0)throw new Error(`workgroupSize.x * workgroupSize.y must be a power of two. (current: ${this.threadsPerWorkgroup})`);this.shaderModules.prefixSum=this.device.createShaderModule({label:"prefix-sum",code:s?z:N}),this.createPassRecursive(t,e)}createPassRecursive(r,e){const i=Math.ceil(e/this.itemsPerWorkgroup),t=P(this.device,i),s=this.device.createBuffer({label:"prefix-sum-block-sum",size:i*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),u=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),o=this.device.createBindGroup({label:"prefix-sum-bind-group",layout:u,entries:[{binding:0,resource:{buffer:r}},{binding:1,resource:{buffer:s}}]}),n=this.device.createPipelineLayout({bindGroupLayouts:[u]}),l=this.device.createComputePipeline({label:"prefix-sum-scan-pipeline",layout:n,compute:{module:this.shaderModules.prefixSum,entryPoint:"reduce_downsweep",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ITEMS_PER_WORKGROUP:this.itemsPerWorkgroup,ELEMENT_COUNT:e}}});if(this.pipelines.push({pipeline:l,bindGroup:o,dispatchSize:t}),i>1){this.createPassRecursive(s,i);const p=this.device.createComputePipeline({label:"prefix-sum-add-block-pipeline",layout:n,compute:{module:this.shaderModules.prefixSum,entryPoint:"add_block_sums",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:e}}});this.pipelines.push({pipeline:p,bindGroup:o,dispatchSize:t})}}getDispatchChain(){return this.pipelines.flatMap(r=>[r.dispatchSize.x,r.dispatchSize.y,1])}dispatch(r,e,i=0){this.pipelines.forEach(({pipeline:t,bindGroup:s,dispatchSize:u},o)=>{r.setPipeline(t),r.setBindGroup(0,s),e?r.dispatchWorkgroupsIndirect(e,i+o*3*4):r.dispatchWorkgroups(u.x,u.y,1)})}}const H=(a=!1,r=!1,e="full")=>`
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
  ${a?Z:"s_data[TID] = select(0u, input[GID], GID < ELEMENT_COUNT);"}

  // Perform parallel reduction
  for (var d = 1u; d < THREADS_PER_WORKGROUP; d *= 2u) {
    workgroupBarrier();  
    if (TID % (2u * d) == 0u) {
      s_data[TID] += s_data[TID + d];
    }
  }
  workgroupBarrier();

  // Write reduction result
  ${r?F(e):Y}
}`,Y=`
  if (TID == 0) {
    output[WORKGROUP_ID] = s_data[0];
  }
`,Z=`
  let LAST_THREAD = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1;

  // Load current element into shared memory
  // Also load next element for comparison
  let elm = select(0u, input[GID], GID < ELEMENT_COUNT);
  let next = select(0u, input[GID + 1], GID < ELEMENT_COUNT-1);
  s_data[TID] = elm;
  workgroupBarrier();

  s_data[TID] = select(0u, 1u, GID < ELEMENT_COUNT-1 && elm > next);
`,F=a=>`
  let fullDispatchLength = arrayLength(&output);
  let dispatchIndex = TID * 3;

  if (dispatchIndex >= fullDispatchLength) {
    return;
  }

  ${a=="full"?$:V}
`,V=`
  output[dispatchIndex] = select(0, original[dispatchIndex], s_data[0] == 0 && is_sorted == 0u);
`,$=`
  if (TID == 0 && s_data[0] == 0) {
    is_sorted = 1u;
  }

  output[dispatchIndex] = select(0, original[dispatchIndex], s_data[0] != 0);
`;class R extends G{constructor({device:e,count:i,workgroupSize:t={x:16,y:16},data:s,result:u,original:o,isSorted:n,start:l=0,mode:p="full"}){super({device:e,count:i,workgroupSize:t});f(this,"start");f(this,"mode");f(this,"buffers",{});f(this,"outputs",[]);this.start=l,this.mode=p,this.buffers={data:s,result:u,original:o,isSorted:n},this.createPassesRecursive(s,i)}static findOptimalDispatchChain(e,i,t){const s=t.x*t.y,u=[];do{const o=Math.ceil(i/s),n=P(e,o);u.push(n.x,n.y,1),i=o}while(i>1);return u}createPassesRecursive(e,i,t=0){const s=Math.ceil(i/this.threadsPerWorkgroup),u=t===0,o=s<=1,n=`check-sort-${this.mode}-${t}`,l=o?this.buffers.result:this.device.createBuffer({label:n,size:s*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),p=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},...o?[{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),m=this.device.createBindGroup({layout:p,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:l}},...o?[{binding:2,resource:{buffer:this.buffers.original}},{binding:3,resource:{buffer:this.buffers.isSorted}}]:[]]}),g=this.device.createPipelineLayout({bindGroupLayouts:[p]}),_=u?this.start+i:i,O=u?this.start:0,E=this.device.createComputePipeline({layout:g,compute:{module:this.device.createShaderModule({label:n,code:H(u,o,this.mode)}),entryPoint:this.mode=="reset"?"reset":"check_sort",constants:{ELEMENT_COUNT:_,WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,...this.mode!=="reset"&&{THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,START_ELEMENT:O}}}});this.outputs.push(l),this.pipelines.push({pipeline:E,bindGroup:m}),o||this.createPassesRecursive(l,s,t+1)}dispatch(e,i,t=0){this.pipelines.forEach(({pipeline:s,bindGroup:u},o)=>{const n=this.mode!=="reset"&&(this.mode==="full"||o<this.pipelines.length-1);e.setPipeline(s),e.setBindGroup(0,u),n&&i?e.dispatchWorkgroupsIndirect(i,t+o*3*4):e.dispatchWorkgroups(1,1,1)})}}const X=`
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
`,q=`
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
`,x=`
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
`;var T,C,U,W;class Q extends G{constructor({device:e,count:i,workgroupSize:t={x:16,y:16},texture:s,keys:u,values:o,bitCount:n=32,checkOrder:l=!1,localShuffle:p=!1,avoidBankConflicts:m=!1}){super({device:e,count:i,workgroupSize:t});k(this,T);k(this,U);f(this,"bitCount");f(this,"checkOrder",!1);f(this,"localShuffle",!1);f(this,"avoidBankConflicts",!1);f(this,"prefixBlockWorkgroupCount");f(this,"hasValues");f(this,"dispatchSize",{x:1,y:1});f(this,"dispatchOffsets",{radixSort:0,checkSortFast:3*4,prefixSum:6*4});f(this,"initialDispatch",[]);f(this,"kernels",{});f(this,"buffers",{});f(this,"texture");if(!e)throw new Error("No device provided");if(!u&&!s)throw new Error("No keys buffer or texture provided");if(!Number.isInteger(i)||i<=0)throw new Error("Invalid count parameter");if(!Number.isInteger(n)||n<=0||n>32)throw new Error(`Invalid bitCount parameter: ${n}`);if(!Number.isInteger(t.x)||!Number.isInteger(t.y))throw new Error("Invalid workgroupSize parameter");if(n%4!=0)throw new Error("bitCount must be a multiple of 4");this.bitCount=n,this.checkOrder=l,this.localShuffle=p,this.avoidBankConflicts=m,this.prefixBlockWorkgroupCount=4*this.workgroupCount,this.hasValues=!!o||!!s,this.texture=s,this.buffers={keys:u,values:o},this.createShaderModules(),this.createPipelines()}createShaderModules(){const e=t=>t.split(`
`).filter(s=>!s.toLowerCase().includes("values")).join(`
`),i=this.localShuffle?q:X;this.shaderModules={blockSum:this.device.createShaderModule({label:"radix-sort-block-sum",code:this.hasValues?i:e(i)}),reorder:this.device.createShaderModule({label:"radix-sort-reorder",code:this.hasValues?x:e(x)})}}createPipelines(){this.createPrefixSumKernel();const e=this.calculateDispatchSizes();this.createBuffers(e),this.createCheckSortKernels(e);for(let i=0;i<this.bitCount;i+=2){const t=i%4==0,s=t?this.buffers.keys:this.buffers.tmpKeys,u=t?this.buffers.values:this.buffers.tmpValues,o=t?this.buffers.tmpKeys:this.buffers.keys,n=t?this.buffers.tmpValues:this.buffers.values,l=this.createBlockSumPipeline(s,u,i),p=this.createReorderPipeline(s,u,o,n,i);this.pipelines.push(l,p)}}createPrefixSumKernel(){const e=this.device.createBuffer({label:"radix-sort-prefix-block-sum",size:this.prefixBlockWorkgroupCount*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),i=new A({device:this.device,data:e,count:this.prefixBlockWorkgroupCount,workgroupSize:this.workgroupSize,avoidBankConflicts:this.avoidBankConflicts});this.kernels.prefixSum=i,this.buffers.prefixBlockSum=e}calculateDispatchSizes(){const e=P(this.device,this.workgroupCount),i=this.kernels.prefixSum.getDispatchChain(),t=Math.min(this.count,this.threadsPerWorkgroup*4),s=this.count-t,u=t-1,o=R.findOptimalDispatchChain(this.device,t,this.workgroupSize),n=R.findOptimalDispatchChain(this.device,s,this.workgroupSize),l=[e.x,e.y,1,...o.slice(0,3),...i];return this.dispatchOffsets={radixSort:0,checkSortFast:3*4,prefixSum:6*4},this.dispatchSize=e,this.initialDispatch=l,{initialDispatch:l,dispatchSizesFull:n,checkSortFastCount:t,checkSortFullCount:s,startFull:u}}createBuffers(e){const i=this.device.createBuffer({label:"radix-sort-tmp-keys",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),t=this.hasValues?this.device.createBuffer({label:"radix-sort-tmp-values",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}):void 0,s=this.device.createBuffer({label:"radix-sort-local-prefix-sum",size:this.count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});if(this.buffers.tmpKeys=i,this.buffers.tmpValues=t,this.buffers.localPrefixSum=s,!this.checkOrder)return;const u=S({device:this.device,label:"radix-sort-dispatch-size",data:e.initialDispatch,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.INDIRECT}),o=S({device:this.device,label:"radix-sort-dispatch-size-original",data:e.initialDispatch,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),n=S({label:"check-sort-full-dispatch-size",device:this.device,data:e.dispatchSizesFull,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.INDIRECT}),l=S({label:"check-sort-full-dispatch-size-original",device:this.device,data:e.dispatchSizesFull,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),p=S({label:"is-sorted",device:this.device,data:[0],usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});this.buffers.dispatchSize=u,this.buffers.originalDispatchSize=o,this.buffers.checkSortFullDispatchSize=n,this.buffers.originalCheckSortFullDispatchSize=l,this.buffers.isSorted=p}createCheckSortKernels(e){if(!this.checkOrder)return;const{checkSortFastCount:i,checkSortFullCount:t,startFull:s}=e,u=new R({mode:"full",device:this.device,data:this.buffers.keys,result:this.buffers.dispatchSize,original:this.buffers.originalDispatchSize,isSorted:this.buffers.isSorted,count:t,start:s,workgroupSize:this.workgroupSize}),o=new R({mode:"fast",device:this.device,data:this.buffers.keys,result:this.buffers.checkSortFullDispatchSize,original:this.buffers.originalCheckSortFullDispatchSize,isSorted:this.buffers.isSorted,count:i,workgroupSize:this.workgroupSize}),n=this.initialDispatch.length/3;if(o.threadsPerWorkgroup<u.pipelines.length||u.threadsPerWorkgroup<n){console.warn("Warning: workgroup size is too small to enable check sort optimization, disabling..."),this.checkOrder=!1;return}const l=new R({mode:"reset",device:this.device,data:this.buffers.keys,original:this.buffers.originalDispatchSize,result:this.buffers.dispatchSize,isSorted:this.buffers.isSorted,count:n,workgroupSize:P(this.device,n)});this.kernels.checkSort={reset:l,fast:o,full:u}}createBlockSumPipeline(e,i,t){const s=this.device.createBindGroupLayout({label:"radix-sort-block-sum",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:this.localShuffle?"storage":"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},...this.localShuffle&&this.hasValues?[{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),u=this.device.createBindGroup({layout:s,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:this.buffers.localPrefixSum}},{binding:2,resource:{buffer:this.buffers.prefixBlockSum}},...this.localShuffle&&this.hasValues?[{binding:3,resource:{buffer:i}}]:[]]}),o=this.device.createPipelineLayout({bindGroupLayouts:[s]});return{pipeline:this.device.createComputePipeline({label:"radix-sort-block-sum",layout:o,compute:{module:this.shaderModules.blockSum,entryPoint:"radix_sort",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,WORKGROUP_COUNT:this.workgroupCount,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:this.count,CURRENT_BIT:t}}}),bindGroup:u}}createReorderPipeline(e,i,t,s,u){const o=this.device.createBindGroupLayout({label:"radix-sort-reorder",entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},...this.hasValues?[{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]:[]]}),n=this.device.createBindGroup({layout:o,entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:{buffer:t}},{binding:2,resource:{buffer:this.buffers.localPrefixSum}},{binding:3,resource:{buffer:this.buffers.prefixBlockSum}},...this.hasValues?[{binding:4,resource:{buffer:i}},{binding:5,resource:{buffer:s}}]:[]]}),l=this.device.createPipelineLayout({bindGroupLayouts:[o]});return{pipeline:this.device.createComputePipeline({label:"radix-sort-reorder",layout:l,compute:{module:this.shaderModules.reorder,entryPoint:"radix_sort_reorder",constants:{WORKGROUP_SIZE_X:this.workgroupSize.x,WORKGROUP_SIZE_Y:this.workgroupSize.y,WORKGROUP_COUNT:this.workgroupCount,THREADS_PER_WORKGROUP:this.threadsPerWorkgroup,ELEMENT_COUNT:this.count,CURRENT_BIT:u}}}),bindGroup:n}}dispatch(e){this.checkOrder?v(this,U,W).call(this,e):v(this,T,C).call(this,e)}}T=new WeakSet,C=function(e){for(let i=0;i<this.bitCount/2;i+=1){const t=this.pipelines[i*2],s=this.pipelines[i*2+1];e.setPipeline(t.pipeline),e.setBindGroup(0,t.bindGroup),e.dispatchWorkgroups(this.dispatchSize.x,this.dispatchSize.y,1),this.kernels.prefixSum.dispatch(e),e.setPipeline(s.pipeline),e.setBindGroup(0,s.bindGroup),e.dispatchWorkgroups(this.dispatchSize.x,this.dispatchSize.y,1)}},U=new WeakSet,W=function(e){this.kernels.checkSort.reset.dispatch(e);for(let i=0;i<this.bitCount/2;i++){const t=this.pipelines[i*2],s=this.pipelines[i*2+1];i%2==0&&(this.kernels.checkSort.fast.dispatch(e,this.buffers.dispatchSize,this.dispatchOffsets.checkSortFast),this.kernels.checkSort.full.dispatch(e,this.buffers.checkSortFullDispatchSize)),e.setPipeline(t.pipeline),e.setBindGroup(0,t.bindGroup),e.dispatchWorkgroupsIndirect(this.buffers.dispatchSize,this.dispatchOffsets.radixSort),this.kernels.prefixSum.dispatch(e,this.buffers.dispatchSize,this.dispatchOffsets.prefixSum),e.setPipeline(s.pipeline),e.setBindGroup(0,s.bindGroup),e.dispatchWorkgroupsIndirect(this.buffers.dispatchSize,this.dispatchOffsets.radixSort)}};function y(a,r,e=0){const i=a.createBuffer({size:r.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|e,mappedAtCreation:!0});new Uint32Array(i.getMappedRange()).set(r),i.unmap();const t=a.createBuffer({size:r.length*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});return[i,t]}function j(a){const e=a.createQuerySet({type:"timestamp",count:2}),i=a.createBuffer({size:8*2,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),t=a.createBuffer({size:8*2,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});return{descriptor:{timestampWrites:{querySet:e,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}},resolve:o=>{o.resolveQuerySet(e,0,2,i,0),o.copyBufferToBuffer(i,0,t,0,8*2)},getTimestamps:async()=>{await t.mapAsync(GPUMapMode.READ);const o=new BigUint64Array(t.getMappedRange().slice());return t.unmap(),o}}}const c={elementCount:2**20,bitCount:32,workgroupSize:16,checkOrder:!1,localShuffle:!1,avoidBankConflicts:!1,sortMode:"Keys",initialSort:"Random",consecutiveSorts:1};window.onload=async function(){var t;const r=await((t=navigator.gpu)==null?void 0:t.requestAdapter()),e=await(r==null?void 0:r.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:{maxComputeInvocationsPerWorkgroup:32*32}}));if(!e)throw document.getElementById("info").innerHTML="WebGPU doesn't appear to be supported on this device.",document.getElementById("info").style.color="#ff7a48",new Error("Could not create WebGPU device");const i=new te(e);i.onClickSort=()=>ee(e)};async function J(a,r=!0){const e=new Uint32Array(c.elementCount),i=2**c.bitCount;switch(c.initialSort){case"Random":e.forEach((g,_)=>e[_]=Math.floor(Math.random()*i));break;case"Sorted":e.forEach((g,_)=>e[_]=_);break}const[t]=y(a,e);let s=null;if(c.sortMode==="Keys & Values"){const g=new Uint32Array(c.elementCount).map(()=>Math.floor(Math.random()*1e6));s=y(a,g)[0]}const u=new Q({device:a,keys:t,values:s,count:c.elementCount,bitCount:c.bitCount,workgroupSize:{x:c.workgroupSize,y:c.workgroupSize},checkOrder:c.checkOrder,localShuffle:c.localShuffle,avoidBankConflicts:c.avoidBankConflicts}),o=j(a),n=a.createCommandEncoder(),l=n.beginComputePass(o.descriptor);u.dispatch(l),l.end(),o.resolve(n),a.queue.submit([n.finish()]);const p=await o.getTimestamps(),m={cpu:0,gpu:Number(p[1]-p[0])/1e6};if(r){const g=performance.now();e.sort((_,O)=>_-O),m.cpu=performance.now()-g}return m}async function ee(a){let r=0,e=0;const i=document.getElementById("results");i.children[0].id==="info"&&(i.innerHTML="");const t=d(i,"div","result");t.innerHTML+=`[${i.children.length}] `,t.innerHTML+=`Sorting ${b(D(c.elementCount),"#ff9933")} ${c.sortMode.toLowerCase()} of ${b(c.bitCount.toString(),"#ff9933")} bits`,t.innerHTML+=`<br>Initial sort: ${c.initialSort}, Workgroup size: ${c.workgroupSize}x${c.workgroupSize}`,t.innerHTML+=`<br>Optimizations: (${c.checkOrder}, ${c.localShuffle}, ${c.avoidBankConflicts})`;for(let u=0;u<c.consecutiveSorts;u+=1){const o=await J(a,u==0);r+=o.cpu,e+=o.gpu}const s=e/c.consecutiveSorts;t.innerHTML+=`<br>> CPU Reference: ${b(r.toFixed(2)+"ms","#abff33")}, `,t.innerHTML+=`GPU Average (${b(c.consecutiveSorts.toString(),"#ff9933")} sorts): ${b(s.toFixed(2)+"ms","#abff33")}, `,t.innerHTML+=`Speedup: ${b("x"+(r/s).toFixed(2),r/s>=1?"#abff33":"#ff3333")}<br><br>`,t.scrollIntoView({behavior:"smooth",block:"end"})}class te{constructor(r){f(this,"dom");f(this,"onClickSort",()=>{});this.dom=document.getElementById("gui"),this.createHeader(),this.createTitle("Radix Sort Kernel"),this.createSlider(c,"elementCount","Element Count",c.elementCount,1e4,2**24,1,{logarithmic:!0}),this.createSlider(c,"bitCount","Bit Count",c.bitCount,4,32,4),this.createSlider(c,"workgroupSize","Workgroup Size",c.workgroupSize,2,5,1,{power_of_two:!0}),this.createTitle("Optimizations"),this.createCheckbox(c,"checkOrder","Check If Sorted",c.checkOrder),this.createCheckbox(c,"localShuffle","Local Shuffle",c.localShuffle),this.createCheckbox(c,"avoidBankConflicts","Avoid Bank Conflicts",c.avoidBankConflicts),this.createTitle("Testing"),this.createDropdown(c,"initialSort","Initial Sort",["Random","Sorted"]),this.createDropdown(c,"sortMode","Sort Mode",["Keys","Keys & Values"]),this.createSlider(c,"consecutiveSorts","Consecutive Sorts",c.consecutiveSorts,1,20,1),this.createButton("Run Radix Sort",()=>this.onClickSort(r)),this.addHints()}addHints(){const r=this.dom.querySelectorAll(".gui-ctn"),e=["","Number of elements to sort","Number of bits to sort","Workgroup size in x and y dimensions","","Check if the data is sorted after each pass to stop the kernel early","Use local shuffle optimization (does not seem to improve performance)","Avoid bank conflicts in shared memory (does not seem to improve performance)","","Initial order of the elements","Whether to use values in addition to keys","Number of consecutive sorts to run","Run the Radix Sort kernel !"];r.forEach((i,t)=>{i.title=e[t]})}createTitle(r){const e=d(this.dom,"div","gui-ctn");d(e,"div","gui-title",{innerText:r})}createHeader(){const r=d(this.dom,"div","gui-header"),e=d(r,"div","gui-icon"),i=d(r,"a","gui-link",{href:"https://github.com/kishimisu/WebGPU-Radix-Sort",innerText:"https://github.com/kishimisu/WebGPU-Radix-Sort"}),t='<svg role="img" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><title>GitHub</title><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>';e.innerHTML=t,i.target="_blank"}createSlider(r,e,i,t=0,s=0,u=1,o=.01,{logarithmic:n=!1,power_of_two:l=!1}={}){const p=Math.log(s),g=(Math.log(u)-p)/(u-s),_=h=>n?Math.round(Math.exp(p+g*(h-s))):l?Math.pow(2,h):h,O=h=>n?(Math.log(h)-p)/g+s:l?Math.log2(h):h,E=d(this.dom,"div","gui-ctn");d(E,"label","gui-label",{innerText:i});const w=d(E,"div","gui-input"),I=d(w,"input","gui-slider",{type:"range",min:s,max:u,step:o,value:O(t)}),M=d(w,"span","gui-value",{innerText:D(t)});I.addEventListener("input",()=>{let h=parseFloat(I.value);(n||l)&&(h=_(parseFloat(I.value))),r[e]=h,M.innerText=D(h)})}createCheckbox(r,e,i,t=!1){const s=d(this.dom,"div","gui-ctn");d(s,"label","gui-label",{innerText:i});const u=d(s,"div","gui-input"),o=d(u,"input","gui-checkbox",{type:"checkbox",checked:t}),n=d(u,"span","gui-value",{innerText:t?"true":"false"});o.addEventListener("change",()=>{r[e]=o.checked,n.innerText=o.checked?"true":"false"})}createDropdown(r,e,i,t){const s=d(this.dom,"div","gui-ctn");d(s,"label","gui-label",{innerText:i});const u=d(s,"div","gui-input"),o=d(u,"select","gui-select"),n=d(u,"span","gui-value",{innerText:r[e]});t.forEach((l,p)=>{const m=d(o,"option","",{innerText:l,value:p});l===r[e]&&(m.selected=!0)}),o.addEventListener("change",()=>{r[e]=t[parseInt(o.value)],n.innerText=t[parseInt(o.value)]})}createButton(r,e){const i=d(this.dom,"div","gui-ctn");d(i,"button","gui-button",{innerText:r}).addEventListener("click",e)}}function d(a,r,e,i={}){const t=document.createElement(r);return t.className=e,Object.keys(i).forEach(s=>{t[s]=i[s]}),a.appendChild(t),t}const b=(a,r)=>`<span style="color:${r}">${a}</span>`,D=a=>a.toString().replace(/\B(?=(\d{3})+(?!\d))/g,",");
