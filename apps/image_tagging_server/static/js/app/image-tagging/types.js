/**
 * @typedef {import('@canvas_types').CanvasState} CanvasState
 * @typedef {import('/app-static/js/app/image-tagging/canvas/groups').CanvasBboxGroup} CanvasBboxGroup
 * @typedef {import('@web_api').RuntimeImage} RuntimeImage
 */

/**
 * @typedef {Object} ImageTaggingState
 * @property {CanvasState} canvas
 * @property {string} currentTool
 * @property {string|null} currentLabelUuid
 * @property {string|null} selectedBoxUuid
 * @property {import('@ui_types').FileNavigation} fileNavigation
 * @property {RuntimeImage|null} image
 * @property {(uuid:string) => CanvasBboxGroup|undefined} getBboxGroup
 * @property {(tool:string) => void} setTool
 * @property {(uuid:string) => void} setLabelUuid
 * @property {(image:RuntimeImage) => void} setImage
 * @property {(bboxGroup:CanvasBboxGroup) => void} removeBboxGroup
 * @property {(bboxGroup:CanvasBboxGroup) => void} updateCanvasBbox
 * @property {(uuid:string) => void} setSelectedBboxUuid
 * @property {(t:import('@konva').Transformer) => void} setTransformer
 * @property {() => void} clearBboxes
 * @property {() => void} clearSelection
 * @property {(container:HTMLElement) => void} initCanvas
 */

export { };