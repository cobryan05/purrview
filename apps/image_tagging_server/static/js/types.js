/**
 * @typedef {import('@konva').default.Stage} Stage
 * @typedef {import('@konva').default.Layer} Layer
 * @typedef {import('@konva').default.Transformer} Transformer
 * @typedef {import('@konva').default.Group} Group
 * @typedef {import('@web_api').ImageMetadata} ImageMetadata
 * @typedef {import('/app-static/js/app/image-tagging/canvas/groups').CanvasBboxGroup} CanvasBboxGroup
 */


/**
 * @typedef {Object} BboxMetadata
 * @property {string} uuid


/**
 * @typedef {Object} RuntimeBboxInfo
 * @property {string} uuid
 * @property {number} x
 * @property {number} y
 * @property {number} width
 * @property {number} height
 * @property {number} [confidence]
 * @property {string} [text]
 * @property {string|null} classUuid
 * @property {string[]} [tagUuids]
 */

/**
 * @typedef {Object} RuntimeImage
 * @property {string|null} name
 * @property {HTMLImageElement|null} img
 * @property {Map<string,RuntimeBboxInfo>|null} bboxes
 */

/**
 * @typedef {Object} CanvasState
 * @property {Stage|null} stage
 * @property {Layer|null} layer
 * @property {Transformer|null} transformer
 * @property {Map<string,CanvasBboxGroup>} bboxes
 */

/**
 * @typedef {Object} InferenceResult
 * @property {RuntimeBboxInfo[]} detections
 */

/**
 * @typedef {Object} State
 * @property {CanvasState} canvas
 * @property {string} currentTool
 * @property {string|null} currentClassUuid
 * @property {string|null} selectedBoxUuid
 * @property {RuntimeImage|null} image
 * @property {string} selectedTool
 * @property {string|null} selectedBboxUuid
 * @property {string|null} classUuid
 * @property {(uuid:string) => CanvasBboxGroup|undefined} getBboxGroup
 * @property {(tool:string) => void} setTool
 * @property {(uuid:string) => void} setClassUuid
 * @property {(image:RuntimeImage) => void} setImage
 * @property {(bboxGroup:CanvasBboxGroup) => void} removeBboxGroup
 * @property {(bboxGroup:CanvasBboxGroup) => void} updateCanvasBbox
 * @property {(bboxUuid:string) => void} setSelectedBboxUuid
 * @property {(t:Transformer) => void} setTransformer
 * @property {() => void} clearBboxes
 * @property {() => void} clearSelection
 * @property {(container:HTMLElement) => void} initStage
 */

export { };