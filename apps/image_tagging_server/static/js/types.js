/**
 * @typedef {import('@konva').default.Stage} Stage
 * @typedef {import('@konva').default.Layer} Layer
 * @typedef {import('@konva').default.Transformer} Transformer
 * @typedef {import('@konva').default.Group} Group
 */

/**
 * @typedef {Object} BBoxInfo
 * @property {number} x
 * @property {number} y
 * @property {number} width
 * @property {number} height
 * @property {number} confidence
 * @property {string} [text]
 * @property {string} [classUuid]
 * @property {string[]} [tagUuids]
 */

/**
 * @typedef {Object} ImageState
 * @property {string|null} name
 * @property {HTMLImageElement|ImageBitmap|null} data
 * @property {Map<string,BBoxInfo>|null} bboxes
 */

/**
 * @typedef {Object} CanvasState
 * @property {Stage|null} stage
 * @property {Layer|null} layer
 * @property {Transformer|null} transformer
 * @property {Map<string,Group>|null} bboxes
 */

/**
 * @typedef {Object} InferenceResult
 * @property {BBoxInfo[]} detections
 */

/**
 * @typedef {Object} State
 * @property {CanvasState} canvas
 * @property {string} currentTool
 * @property {string|null} currentClassUuid
 * @property {string|null} selectedBoxUuid
 * @property {ImageState} image
 * @property {string} selectedTool
 * @property {string|null} selectedBboxUuid
 * @property {string|null} classUuid
 * @property {(uuid:string) => Group|undefined} getBboxGroup
 * @property {(tool:string) => void} setTool
 * @property {(uuid:string) => void} setClassUuid
 * @property {(name:string, img:any, bboxes:Map<string,BBoxInfo>) => void} setImage
 * @property {(group:Group) => void} updateBboxGroup
 * @property {(bboxUuid:string) => void} setSelectedBboxUuid
 * @property {(t:Transformer) => void} setTransformer
 * @property {() => void} clearBboxes
 * @property {() => void} clearSelection
 * @property {(container:HTMLElement) => void} initStage
 */

export {};