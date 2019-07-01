const crypto = require('crypto');
const fs = require('fs');

class RandomNumberPool {
    constructor() {
        this.size = 1024;
        this.array = new Uint32Array(this.size);
        this.offset = 0;
        crypto.randomFillSync(this.array);
    }

    next() {
        if (this.offset === this.size) {
            crypto.randomFillSync(this.array);
            this.offset = 0;
        }
        return this.array[this.offset++];
    }
}

function combineIndex(index1, index2, orderInsensitive = false) {
    if (orderInsensitive) {
        if (index1 > index2) {
            [index1, index2] = [index2, index1];
        }
    }
    return (index1 + index2) * (index1 + index2 + 1) / 2 + index1;
}

function generatePoints(rnp, sideLength, distinguishability, number) {
    const points = [];

    const center = { x: sideLength / 2, y: sideLength / 2 };
    const radius2 = Math.pow(Math.min(sideLength / 2, sideLength / 2), 2);

    const pointList = new Set();
    while (points.length < number) {
        let x = rnp.next() % sideLength;
        let y = rnp.next() % sideLength;

        if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) > radius2) {
            continue;
        }

        const pointIndex = combineIndex(Math.floor(x / distinguishability), Math.floor(y / distinguishability));
        if (pointList.has(pointIndex)) {
            continue;
        }

        points.push({ x: x / sideLength, y: y / sideLength });
        pointList.add(pointIndex);
    }

    return points;
}

function anticlockwise(p1, p2, p3) {
    return Math.sign((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y));
}

function aligned(p1, p2, p3) {
    return Math.sign((p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y));
}

function normalizeTriangle(pointList, indices) {
    if (anticlockwise(pointList[indices[0]], pointList[indices[1]], pointList[indices[2]]) !== 1) {
        [indices[1], indices[2]] = [indices[2], indices[1]];
    }
    return indices;
}

function tangentRange(pointList, pointIndices, sourcePointIndex, scope) {
    if (!scope) {
        scope = [0, pointIndices.length];
    }
    let range = [scope[0], scope[0]];
    for (let i = scope[0]; i < (scope[0] > scope[1] ? scope[1] + pointIndices.length : scope[1]); i++) {
        const direction = anticlockwise(
            pointList[pointIndices[i % pointIndices.length]],
            pointList[pointIndices[(i + 1) % pointIndices.length]],
            pointList[sourcePointIndex]);
        if (direction === -1 || direction === 0 && aligned(
            pointList[sourcePointIndex],
            pointList[pointIndices[i % pointIndices.length]],
            pointList[pointIndices[(i + 1) % pointIndices.length]]) < 0) {
            if (range[0] === range[1]) {
                range = [i, i + 1];
            } else if (range[1] === i) {
                range[1] = i + 1;
            } else if (range[0] === scope[0]) {
                range[0] = i;
                range[1] = range[1] + pointIndices.length;
            }
        }
    }

    return [range[0] % pointIndices.length, range[1] % pointIndices.length];
}

function findConvexHullIndices(pointList) {
    if (pointList.length === 1) {
        return [0];
    }

    const pointIndices = [...Array(pointList.length).keys()];

    let j = 2;
    while (j < pointList.length && anticlockwise(pointList[0], pointList[1], pointList[j]) === 0) {
        j++;
    }

    if (j === pointList.length) { //All points in same line
        if (pointList[pointIndices[0]].x === pointList[pointIndices[1]].x) {
            pointIndices.sort((a, b) => pointList[a].y - pointList[b].y);
        } else {
            pointIndices.sort((a, b) => pointList[a].x - pointList[a].x);
        }

        const convexHullPointIndices = [...pointIndices];
        for (let i = pointIndices.length - 2; i > 0; i--) {
            convexHullPointIndices.push(pointIndices[i]);
        }
        return convexHullPointIndices;
    } else {
        const initialPointList = [...pointIndices.splice(j, 1), ...pointIndices.splice(0, 2)];

        const convexHullPointIndices = normalizeTriangle(pointList, initialPointList);
        while (pointIndices.length !== 0) {
            const pointIndex = pointIndices.shift();
            const range = tangentRange(pointList, convexHullPointIndices, pointIndex);
            if (range[0] < range[1]) {
                convexHullPointIndices.splice(range[0] + 1, range[1] - range[0] - 1, pointIndex);
            } else if (range[0] > range[1]) {
                convexHullPointIndices.splice(range[0] + 1);
                convexHullPointIndices.splice(0, range[1], pointIndex);
            }
        }

        return convexHullPointIndices;
    }
}

function generate(fileName, maxPointNumber, quantityForEachPointNumber, sideLength, distinguishability) {
    fs.writeFileSync(fileName, '', { flag: 'w' });

    const rnp = new RandomNumberPool();

    for (let pointNumber = 5; pointNumber <= maxPointNumber; pointNumber++) {

        for (let i = 0; i < quantityForEachPointNumber; i++) {
            const pointList = generatePoints(rnp, sideLength, distinguishability, pointNumber);
            const convexHullPointIndices = findConvexHullIndices(pointList);

            const minIndex = Math.min(...convexHullPointIndices);
            while (convexHullPointIndices[0] !== minIndex) {
                convexHullPointIndices.push(convexHullPointIndices.shift());
            }
            convexHullPointIndices.push(convexHullPointIndices[0]);

            const line = `${pointList.reduce((a, b) => a.concat(b.x, b.y), []).join(' ')} output ${convexHullPointIndices.map(i => i + 1).join(' ')}\n`;
            fs.writeFileSync(fileName, line, { flag: 'a' });
        }

        console.info(`Generating for point number (${pointNumber}) is done.`);
    }
}

generate('data/ch_all_training.txt', 50, 20000, 128, 16);
generate('data/ch_all_test.txt', 50, 1000, 128, 16);