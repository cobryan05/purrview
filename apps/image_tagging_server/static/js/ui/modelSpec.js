/**
 * Model type / architecture catalogue and node ↔ UI-state conversion helpers
 * for hierarchical model configs. Pure data + functions, no DOM.
 *
 * A model config's `nodes` array encodes the sub-model hierarchy: one root
 * node (labelUuid=null) plus one node per assigned label; a label node whose
 * spec differs from its inherited spec is a sub-model boundary.
 */

/**
 * Returns the label scope that a given model type operates on.
 * image_classifier works on whole-image labels; everything else uses object/bbox labels.
 * @param {string} modelType
 * @returns {'image' | 'object'}
 */
export function scopeForModelType(modelType) {
  return modelType === 'image_classifier' ? 'image' : 'object';
}

export const MODEL_SPEC = {
  object_detection: {
    label: 'Object Detection',
    architectures: {
      yolo_v9_t: 'YOLOv9-t  (~2M params, ~8 MB)',
      yolo_v9_s: 'YOLOv9-s  (~7M params, ~26 MB)',
      yolo_v9_m: 'YOLOv9-m  (~20M params, ~76 MB)',
      yolo_v9_c: 'YOLOv9-c  (~25M params, ~102 MB)',
    },
  },
  crop_classifier: {
    label: 'Crop Classifier',
    architectures: {
      efficientnet_b0: 'EfficientNet-B0',
      efficientnet_b2: 'EfficientNet-B2',
      efficientnet_b4: 'EfficientNet-B4',
      mobilenet_v3_small: 'MobileNetV3-small',
      mobilenet_v3_large: 'MobileNetV3-large',
      resnet_18: 'ResNet-18',
      resnet_50: 'ResNet-50',
    },
  },
  image_classifier: {
    label: 'Image Classifier',
    architectures: {
      efficientnet_b0: 'EfficientNet-B0',
      efficientnet_b2: 'EfficientNet-B2',
      efficientnet_b4: 'EfficientNet-B4',
      mobilenet_v3_small: 'MobileNetV3-small',
      mobilenet_v3_large: 'MobileNetV3-large',
      resnet_18: 'ResNet-18',
      resnet_50: 'ResNet-50',
    },
  },
};

/** Short display string for a spec, e.g. "Object Detection · YOLOv9-s". */
export function specDisplay({ modelType, architecture } = {}) {
  const typeLabel = MODEL_SPEC[modelType]?.label ?? modelType ?? '?';
  const archLabel = (MODEL_SPEC[modelType]?.architectures?.[architecture] ?? architecture ?? '?')
    .split('(')[0].trim(); // drop the "(~7M params…)" size hint in compact contexts
  return `${typeLabel} · ${archLabel}`;
}

/**
 * Walk up the label tree to find the nearest ancestor with a submodel.
 * Falls back to rootSpec.
 */
export function inheritedModelSpec(uuid, labelsMap, submodels, rootSpec) {
  let p = labelsMap.get(uuid)?.parentUuid ?? null;
  while (p !== null) {
    if (submodels.has(p)) return submodels.get(p);
    p = labelsMap.get(p)?.parentUuid ?? null;
  }
  return rootSpec;
}

/**
 * Parse a flat nodes array from the API into form UI state.
 * Processes ancestors first so inheritance is resolved correctly.
 *
 * @returns {{ rootSpec: {modelType:string,architecture:string}, selected: Set<string>,
 *             submodels: Map<string,{modelType:string,architecture:string}>,
 *             rollUpUnchecked: Set<string> }}
 */
export function parseNodes(nodes, labelsMap) {
  const defaultType = Object.keys(MODEL_SPEC)[0];
  const defaultArch = Object.keys(MODEL_SPEC[defaultType].architectures)[0];
  const defaultSpec = { modelType: defaultType, architecture: defaultArch };

  let rootSpec = null;
  const nonRoot = [];

  for (const n of (nodes ?? [])) {
    if (n.labelUuid === null) {
      rootSpec = { modelType: n.modelType, architecture: n.architecture };
    } else {
      nonRoot.push(n);
    }
  }
  rootSpec = rootSpec ?? defaultSpec;

  const depthOf = (uuid) => {
    let d = 0, p = labelsMap.get(uuid)?.parentUuid ?? null;
    while (p) { d++; p = labelsMap.get(p)?.parentUuid ?? null; }
    return d;
  };
  nonRoot.sort((a, b) => depthOf(a.labelUuid) - depthOf(b.labelUuid));

  const submodels = new Map();
  const selected = new Set();
  const rollUpUnchecked = new Set();

  for (const { labelUuid, modelType, architecture, rollUpUnchecked: rup } of nonRoot) {
    selected.add(labelUuid);
    if (rup) rollUpUnchecked.add(labelUuid);
    const inh = inheritedModelSpec(labelUuid, labelsMap, submodels, rootSpec);
    if (modelType !== inh.modelType || architecture !== inh.architecture) {
      submodels.set(labelUuid, { modelType, architecture });
    }
  }

  return { rootSpec, selected, submodels, rollUpUnchecked };
}

/**
 * Build the nodes array to send to the API.
 * Union of selected labels and submodel-boundary labels.
 */
export function computeNodes(selected, submodels, rollUpUnchecked, rootSpec, labelsMap) {
  const nodes = [{ labelUuid: null, ...rootSpec }];
  const allUuids = new Set([...selected, ...submodels.keys()]);
  for (const uuid of allUuids) {
    const spec = submodels.has(uuid)
      ? submodels.get(uuid)
      : inheritedModelSpec(uuid, labelsMap, submodels, rootSpec);
    nodes.push({ labelUuid: uuid, ...spec, rollUpUnchecked: rollUpUnchecked.has(uuid) });
  }
  return nodes;
}
