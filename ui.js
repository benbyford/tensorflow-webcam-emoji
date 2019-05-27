/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs';

const CONTROLS = ['_1','_2',"_3", "_4", "_5"];
const totals = [0, 0, 0, 0, 0];

const trainStatusElement = document.getElementById('train-status');

// Set hyper params from UI values.
const learningRateElement = document.getElementById('learningRate');
export const getLearningRate = () => +learningRateElement.value;

const batchSizeFractionElement = document.getElementById('batchSizeFraction');
export const getBatchSizeFraction = () => +batchSizeFractionElement.value;

const epochsElement = document.getElementById('epochs');
export const getEpochs = () => +epochsElement.value;

const denseUnitsElement = document.getElementById('dense-units');
export const getDenseUnits = () => +denseUnitsElement.value;

const statusElement = document.getElementById('status');

// export function startPacman() {
//   google.pacman.startGameplay();
// }

export function predictClass(classId) {
    // ***** do somethig with prediction

    // indicate which item has been predicted
    document.body.setAttribute('data-active', CONTROLS[classId]);

    showPrediction(classId);
}

export function isPredicting() {
  statusElement.style.visibility = 'visible';
}
export function donePredicting() {
  statusElement.style.visibility = 'hidden';

}
export function trainStatus(status) {
  trainStatusElement.innerText = status;
}

export let addExampleHandler;
export function setExampleHandler(handler) {
  addExampleHandler = handler;
}
let mouseDown = false;

// show current prediction
const currentPrediction = document.getElementById('current-prediction');
function showPrediction(prediction) {
    currentPrediction.innerText = CONTROLS[prediction];
}

const thumbDisplayed = {};

async function handler(label) {
  mouseDown = true;

  const className = CONTROLS[label];
  const button = document.getElementById(className);
  const total = document.getElementById(className + '-total');
  while (mouseDown) {
    addExampleHandler(label);
    document.body.setAttribute('data-active', CONTROLS[label]);
    total.innerText = ++totals[label];
    await tf.nextFrame();
  }
  document.body.removeAttribute('data-active');
}


const upButton = document.getElementById('_1');
const downButton = document.getElementById('_2');
const leftButton = document.getElementById('_3');
const rightButton = document.getElementById('_4');
const coolButton = document.getElementById('_5');

const btnArray = [upButton,downButton,leftButton,rightButton,coolButton];

upButton.addEventListener('mousedown', () => handler(0));
upButton.addEventListener('mouseup', () => mouseDown = false);

downButton.addEventListener('mousedown', () => handler(1));
downButton.addEventListener('mouseup', () => mouseDown = false);

leftButton.addEventListener('mousedown', () => handler(2));
leftButton.addEventListener('mouseup', () => mouseDown = false);

rightButton.addEventListener('mousedown', () => handler(3));
rightButton.addEventListener('mouseup', () => mouseDown = false);

coolButton.addEventListener('mousedown', () => handler(4));
coolButton.addEventListener('mouseup', () => mouseDown = false);

export function drawThumb(img, label) {
  if (thumbDisplayed[label] == null) {
    const thumbCanvas = document.getElementById(CONTROLS[label] + '-thumb');
    draw(img, thumbCanvas);
  }
}

export function draw(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

//👌
const emojiList = ["😀","😁","😂","🤣","😃","😄","😅","😆","😉","😊","😋","😎","😍","😘","😗","😙","😚","🙂","🤗","🤩","🤔","🤨","😐","😑","😶","🙄","😏","😣","😥","😮","🤐","😯","😪","😫","😴","😌","😛","😜","😝","🤤","😒","😓","😔","😕","🙃","🤑","😲","🙁","😖","😞","😟","😤","😢","😭","😦","😧","😨","😩","🤯","😬","😰","😱","😳","🤪","😵","😡","😠","🤬","😷","🤒","🤕","🤢","🤮","🤧","😇","🤠","🤥","🤫","🤭","🧐","🤓"];

const emojis = document.getElementsByClassName('emoji');
for (let i = 0; i < emojis.length; i++) {
    emojis[i].addEventListener('mousedown', emojiPicker);
    emojis[i].addEventListener('mouseup', () => mouseDown = false);
}
function emojiPicker(el){
    
    if(el.target.classList.contains("toggled") || el.target.parentNode.classList.contains("toggled")){
        return;
    }

    el.target.classList.add("toggled");
    
    const div = document.createElement("div");
    div.setAttribute("class","emoji-picker");

    for (let i = 0; i < emojiList.length; i++) {
        
        const a = document.createElement("a");
        a.setAttribute("href", "#"+emojiList[i].toString());
        a.innerText = emojiList[i].toString();
        div.appendChild(a);

        a.addEventListener('mousedown', selectEmoji);
        a.addEventListener('mouseup', () => mouseDown = false);
    };
    
    el.target.appendChild(div);
}

function selectEmoji(el){
    // reset toggle
    el.target.parentNode.parentNode.classList.remove("toggled");

    el.target.parentNode.parentNode.innerHTML = el.target.innerHTML;
}