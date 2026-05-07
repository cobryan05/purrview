/**
 * @typedef {import('@canvas_types').CanvasState} CanvasState
 * @typedef {import('@canvas_types').BboxView} BboxView
 * @typedef {import('@domain_image').UIImage} UIImage
 */

/**
 * @typedef {Object} ImageTaggingState
 * @property {CanvasState} canvas
 * @property {UIImage|null} image
 * @property {string} currentTool
 * @property {string|null} currentLabelUuid
 * @property {string|null} selectedBoxUuid
 * @property {import('@ui_types').FileNavigation} fileNavigation
 * @property {(uuid:string) => BboxView|undefined} getBboxView
 * @property {(tool:string) => void} setTool
 * @property {(uuid:string) => void} setLabelUuid
 * @property {(image:UIImage) => void} setImage
 * @property {(bboxView:BboxView) => void} removeBboxView
 * @property {(bboxView:BboxView) => void} updateCanvasBboxView
 * @property {(uuid:string) => void} setSelectedBboxUuid
 * @property {(t:import('@konva').Transformer) => void} setTransformer
 * @property {() => void} clearBboxes
 * @property {() => void} clearSelection
 * @property {(container:HTMLElement) => void} initCanvas
 */

export { };