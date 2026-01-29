// @ts-check
import { Logger } from '/app-static/js/ui/utils/index.js';

/**
 * @typedef {import('@app_types').AppState} AppState
 */

import { imageTaggingState } from './image-tagging/state.js';

/** @type {AppState} */
export const appState = {
  imageTagging: imageTaggingState,
};
