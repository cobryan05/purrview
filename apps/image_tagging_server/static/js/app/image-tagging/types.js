/**
 * @typedef {import('/app-static/js/app/image-tagging/canvas/views/bboxView.js').BboxView} BboxView
 * @typedef {import('@domain_image').UIImage} UIImage
 */

/**
 * @typedef {Object} CanvasRect
 * @property {number} x
 * @property {number} y
 * @property {number} width
 * @property {number} height
 */

/**
 * @typedef {Object} ViewportTransform
 * @property {number} offsetX
 * @property {number} offsetY
 * @property {number} scale
 * @property {number} imageWidth
 * @property {number} imageHeight
 */


/**
 * @typedef {Object} CanvasState
 * @property {HTMLElement|null} container
 * @property {Stage|null} stage
 * @property {Layer|null} layer
 * @property {Konva.Transformer|null} transformer
 * @property {ViewportTransform|null} viewport
 * @property {Map<string, BboxView>} bboxViews
 * @property {Konva.Image|null} backgroundImage
 */

/**
 * @typedef {Object} ImageTaggingState
 * @property {UIImage|null} image
 * @property {string|null} currentTool
 * @property {string|null} currentLabelUuid
 * @property {string|null} selectedBoxUuid
 */

/**
 * @typedef {Object} ImageTaggingRuntime
 * @property {ImageTaggingState|null} state
 * @property {CanvasState|null} canvas
 *
 * @property {(container:HTMLElement|null) => void} setImageTaggingPane
 * @property {HTMLElement|null} imageTaggingPane
 *
 * @property {(navigation:import('@ui_types').FileNavigation|null) => void} setFileNavigation
 * @property {import('@ui_types').FileNavigation|null} fileNavigation
 *
 * @property {(tool:string) => void} setTool
 * @property {string|null} tool
 * @property {(uuid:string) => void} setLabelUuid
 * @property {string|null} labelUuid
 * @property {(image:UIImage) => void} setImage
 * @property {(uuid:string) => void} setSelectedBboxUuid
 *
 * @property {() => void} clearBboxes
 * @property {() => void} clearSelection
 *
 * @property {(uuid:string) => BboxView|undefined} getBboxView
 */

export { };