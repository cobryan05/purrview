import { createCanvasState } from './canvas/state.js';
import { computeViewport, screenToImage } from './canvas/viewport.js';
import { setCanvasImage, setCanvasViewport } from './canvas/manager.js';
import { Logger } from '/app-static/js/ui/utils/index.js';

/**
 * @returns {import('@image_tagging_types').ImageTaggingRuntime}
 */
export function createImageTaggingRuntime() {
  /** @type {import('@image_tagging_types').ImageTaggingState|null} */
  let _imgTaggingState = { image: null, currentTool: null, currentLabelUuid: null, selectedBoxUuid: null };
  /** @type {import('@image_tagging_types').CanvasState|null} */
  let _canvasState = null;
  /** @type {import('@ui_types').FileNavigation|null} */
  let _fileNavigation = null;
  /** @type {string} */
  let _currentTool = 'select';
  /** @type {HTMLElement | null} */
  let _pane = null;
  /** @type {ViewportTransform|null} */
  let viewport = null;

  /** @param {HTMLElement | null} pane */
  function setImageTaggingPane(pane) {
    if (!pane) return;
    if (_pane) {
      Logger.warn("Replacing existing pane");
      // TODO: Cleanup?
    }
    _pane = pane;
    const stage = _pane.getElementsByClassName("canvas-container")[0];
    _canvasState = createCanvasState(stage);
  }

  /** @param {import('@image_tagging_types').UIImage} image */
  function setImage(image) {
    if (!_canvasState) {
      Logger.warn("No canvas available to display image");
      return;
    }
    if (!_imgTaggingState) {
      Logger.warn("No image tagging state available to set image");
      return;
    }
    _imgTaggingState.image = image;
    setCanvasImage(_canvasState, _imgTaggingState.image);
  }

  /** @param {import('@ui_types').FileNavigation|null} navigation */
  function setFileNavigation(navigation) {
    if (_fileNavigation) {
      Logger.warn("Replacing existing file navigation");
      // TODO: Cleanup?
    }
    _fileNavigation = navigation;
  }

  /** @param {string} tool */
  function setTool(tool) {
    _currentTool = tool;
  }
  function clearBboxes() {
  }


  return {
    get state() { return _imgTaggingState; },
    get canvas() { return _canvasState; },
    get imageTaggingPane() { return _pane; },
    get fileNavigation() { return _fileNavigation; },

    setImageTaggingPane,
    setFileNavigation,
    setTool,
    setImage,
    clearBboxes,
  };
}