const crypto = require('crypto');
const fs = require('fs');

class RandomEngine {
    constructor() {
        this.size = 1024;
        this.array = new Uint32Array(this.size);
        this.refill();
    }

    refill() {
        crypto.randomFillSync(this.array);
        this.offset = 0;
    }

    next() {
        if (this.offset === this.size) {
            this.refill();
        }
        return this.array[this.offset++];
    }
}

function pair(a, b, c = undefined) {
    if (c === undefined) {
        return (a + b) * (a + b + 1) / 2 + b;
    } else {
        return pair(pair(a, b), c);
    }
}

function* randomPoints(randomEngine, pointNumber) {
    const resolution = 1024;
    const grid = 16;
    const padding = 4;

    const center = { x: resolution / 2, y: resolution / 2 };
    const radius2 = Math.pow(Math.min(resolution / 2, resolution / 2), 2);

    const pointList = new Set();
    while (pointList.size < pointNumber) {
        let x = randomEngine.next() % resolution;
        let y = randomEngine.next() % resolution;

        if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) > radius2) {
            continue;
        }

        const pointIndex = pair(Math.floor(x / grid), Math.floor(y / grid));
        if (pointList.has(pointIndex)) {
            continue;
        }

        if (x % grid < padding) {
            x += padding;
        } else if (x % grid >= grid - padding) {
            x -= padding;
        }

        if (y % grid < padding) {
            y += padding;
        } else if (y % grid >= grid - padding) {
            y -= padding;
        }

        yield { x: x / resolution, y: y / resolution };
        pointList.add(pointIndex);
    }
}

function distance(p1, p2) {
    return Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

function aligned(p1, p2, p3) {
    return Math.sign((p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y));
}

function anticlockwise(p1, p2, p3) {
    return Math.sign((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y));
}

function convexHull(points) {
    for (let i = 0; i < points.length; i++) {
        if (anticlockwise(points[i], points[(i + 1) % points.length], points[(i + 2) % points.length]) < 0) {
            return false;
        }
    }

    return true;
}

class VectorLink {
    static DatumPoints = [
        { x: 0.0, y: 1.0 },
        { x: -Math.sin(Math.PI * 2 / 3), y: -0.5 },
        { x: Math.sin(Math.PI * 2 / 3), y: -0.5 }
    ];

    constructor(originPoint) {
        this._originPoint = originPoint;

        this._points = [];
        this._circularLinkList = [];
        this._reversedCircularLinkList = [];

        this.setupDatumPoint();
    }

    aligned(nodeIndex1, nodeIndex2) {
        return aligned(this._originPoint, this._points[nodeIndex1], this._points[nodeIndex2]);
    }

    anticlockwise(nodeIndex1, nodeIndex2) {
        return anticlockwise(this._originPoint, this._points[nodeIndex1], this._points[nodeIndex2]);
    }

    setupDatumPoint() {
        for (let i = 1; i <= VectorLink.DatumPoints.length; i++) {
            this._points[-i] = { x: this._originPoint.x + VectorLink.DatumPoints[i - 1].x, y: this._originPoint.y + VectorLink.DatumPoints[i - 1].y };
            this._circularLinkList[-i] = (-i) % 3 - 1;
            this._reversedCircularLinkList[(-i) % 3 - 1] = -i;
        }
    }

    addNode(nodeIndex, point) {
        if (this._circularLinkList[nodeIndex] !== undefined) {
            throw "Invalid parameter.";
        }

        this._points[nodeIndex] = point;

        for (let i = 1; i <= 3; i++) {
            if (this.anticlockwise(-i, nodeIndex) >= 0 && this.anticlockwise(nodeIndex, (-i) % 3 - 1) > 0) {
                let j = -i;
                while (j !== (-i) % 3 - 1) {
                    const nextNodeIndex = this._circularLinkList[j];

                    if (this.anticlockwise(j, nodeIndex) >= 0 && this.anticlockwise(nodeIndex, nextNodeIndex) > 0) {
                        this._circularLinkList[nodeIndex] = nextNodeIndex;
                        this._circularLinkList[j] = nodeIndex;
                        this._reversedCircularLinkList[nodeIndex] = j;
                        this._reversedCircularLinkList[nextNodeIndex] = nodeIndex;
                        break;
                    }

                    j = this._circularLinkList[j]
                }

                if (j !== (-i) % 3 - 1) {
                    return;
                }
            }

        }

        throw "Unexpected.";
    }

    deleteNode(nodeIndex) {
        if (this._circularLinkList[nodeIndex] === undefined) {
            throw "Invalid parameter.";
        }

        const nextNodeIndex = this._circularLinkList[nodeIndex];
        const preNodeIndex = this._reversedCircularLinkList[nodeIndex];
        this._circularLinkList[preNodeIndex] = nextNodeIndex;
        this._reversedCircularLinkList[nextNodeIndex] = preNodeIndex;

        delete this._reversedCircularLinkList[nodeIndex];
        delete this._circularLinkList[nodeIndex];
        delete this._points[nodeIndex];
    }

    * aheadNodes(startNodeIndex) {
        if (this._circularLinkList[startNodeIndex] === undefined) {
            throw "Invalid parameter.";
        }

        let i = this._circularLinkList[startNodeIndex];
        while (i !== startNodeIndex) {
            if (i >= 0) {
                yield i;
            }
            i = this._circularLinkList[i];
        }
        if (i >= 0) {
            yield i;
        }
    }

    allNodes() {
        return this.aheadNodes(-1);
    }
}

class Scheme {

    constructor(points) {
        this._points = points;
        this._adjacencyVectorLink = points.map((point, i) => new VectorLink(point));
        this._boundingPointList = this.buildConnectionScheme();
        this._assemblies = [];
    }

    distance(index1, index2) {
        return distance(this._points[index1], this._points[index2]);
    }

    aligned(index1, index2, index3) {
        return aligned(this._points[index1], this._points[index2], this._points[index3]);
    }

    anticlockwise(index1, index2, index3) {
        return anticlockwise(this._points[index1], this._points[index2], this._points[index3]);
    }

    convexHull(indices) {
        return convexHull(indices.map(i => this._points[i]));
    }

    collinear(indices) {
        if (indices.length < 3) {
            return true;
        }

        let i = 2;
        while (i < indices.length && this.anticlockwise(indices[0], indices[1], indices[i]) === 0) {
            i++;
        }

        return i === indices.length;
    }

    triangleId(index1, index2, index3) {
        if (index1 < index2 && index1 < index3) {
            return pair(index1, index2, index3);
        } else if (index2 < index1 && index2 < index3) {
            return pair(index2, index3, index1);
        } else {
            return pair(index3, index1, index2);
        }
    }

    normalizeTriangle(trianglePointList) {
        if (this.anticlockwise(...trianglePointList) !== 1) {
            [trianglePointList[1], trianglePointList[2]] = [trianglePointList[2], trianglePointList[1]];
        }
        return trianglePointList;
    }

    tangentRange(pointList, sourcePointIndex, scope) {
        if (!scope) {
            scope = [0, pointList.length];
        }
        let range = [scope[0], scope[0]];
        for (let i = scope[0]; i < (scope[0] > scope[1] ? scope[1] + pointList.length : scope[1]); i++) {
            const direction = this.anticlockwise(pointList[i % pointList.length], pointList[(i + 1) % pointList.length], sourcePointIndex);
            if (direction === -1 || direction === 0 && this.aligned(sourcePointIndex, pointList[i % pointList.length], pointList[(i + 1) % pointList.length]) < 0) {
                if (range[0] === range[1]) {
                    range = [i, i + 1];
                } else if (range[1] === i) {
                    range[1] = i + 1;
                } else if (range[0] === scope[0]) {
                    range[0] = i;
                    range[1] = range[1] + pointList.length;
                }
            }
        }

        return [range[0] % pointList.length, range[1] % pointList.length];
    }

    connect(pointIndex1, pointIndex2) {
        this._adjacencyVectorLink[pointIndex1].addNode(pointIndex2, this._points[pointIndex2]);
        this._adjacencyVectorLink[pointIndex2].addNode(pointIndex1, this._points[pointIndex1]);
    }

    disconnect(pointIndex1, pointIndex2) {
        this._adjacencyVectorLink[pointIndex1].deleteNode(pointIndex2);
        this._adjacencyVectorLink[pointIndex2].deleteNode(pointIndex1);
    }

    findConvexHull(pointList) {
        if (pointList.length === 1) {
            return pointList;
        }

        pointList = [...pointList];

        let j = 2;
        while (j < pointList.length && this.anticlockwise(pointList[0], pointList[1], pointList[j]) === 0) {
            j++;
        }

        if (j === pointList.length) { //All points in same line
            if (this._points[pointList[0]].x === this._points[pointList[1]].x) {
                pointList.sort((a, b) => this._points[a].y - this._points[b].y);
            } else {
                pointList.sort((a, b) => this._points[a].x - this._points[b].x);
            }
            for (let i = 0; i < pointList.length - 1; i++) {
                this.connect(pointList[i], pointList[i + 1]);
            }
            const boundingConvexHullPointList = [...pointList];
            for (let i = pointList.length - 2; i > 0; i--) {
                boundingConvexHullPointList.push(pointList[i]);
            }
            return boundingConvexHullPointList;
        } else {
            const initialPointList = [...pointList.splice(j, 1), ...pointList.splice(0, 2)];

            const boundingConvexHullPointList = this.normalizeTriangle(initialPointList);
            while (pointList.length !== 0) {
                const pointIndex = pointList.shift();
                const range = this.tangentRange(boundingConvexHullPointList, pointIndex);
                if (range[0] < range[1]) {
                    boundingConvexHullPointList.splice(range[0] + 1, range[1] - range[0] - 1, pointIndex);
                } else if (range[0] > range[1]) {
                    boundingConvexHullPointList.splice(range[0] + 1);
                    boundingConvexHullPointList.splice(0, range[1], pointIndex);
                }
            }

            for (let i = 0; i < boundingConvexHullPointList.length; i++) {
                this.connect(boundingConvexHullPointList[i], boundingConvexHullPointList[(i + 1) % boundingConvexHullPointList.length]);
            }

            return boundingConvexHullPointList;
        }
    }

    connectInsideConvexHull(pointList) {
        if (pointList.length <= 3) {
            return;
        }

        for (let i = 0; i < pointList.length - 2; i++) {
            for (let j = 0; j < pointList.length - 3; j++) {
                let k = (j + i + 2) % pointList.length;
                if (i < k) {
                    if (pointList.every(u => u === pointList[i] || u === pointList[k] || this.anticlockwise(pointList[i], pointList[k], u) !== 0)) {
                        //found (i, k)

                        this.connect(pointList[i], pointList[k]);
                        this.connectInsideConvexHull(pointList.slice(i, k + 1));
                        this.connectInsideConvexHull([...pointList.slice(k), ...pointList.slice(0, i + 1)]);
                        return;
                    }
                }
            }
        }
    }

    connectBetweenConvexHulls(outerPointList, innerPointList) {
        if (innerPointList.length === 1) {
            for (const pointIndex of outerPointList) {
                this.connect(pointIndex, innerPointList[0]);
            }
        } else {
            let i = 0;
            while (i < outerPointList.length && this.anticlockwise(innerPointList[0], innerPointList[1], outerPointList[i]) !== -1) {
                i++;
            }

            if (i === outerPointList.length) {
                throw "Unexpected.";
            }

            const range = this.tangentRange(innerPointList, outerPointList[i]);
            if (range[0] < range[1]) {
                for (let j = range[0]; j <= range[1]; j++) {
                    this.connect(outerPointList[i], innerPointList[j]);
                }
            } else if (range[0] > range[1]) {
                for (let j = range[0]; j <= range[1] + innerPointList.length; j++) {
                    this.connect(outerPointList[i], innerPointList[j % innerPointList.length]);
                }
            } else {
                throw "Unexpected.";
            }

            const scope = [range[1], range[0]];
            for (let k = 0; k < outerPointList.length - 1; k++) {
                i = (i + 1) % outerPointList.length;
                const range = this.tangentRange(innerPointList, outerPointList[i], scope);
                if (range[0] === range[1]) {
                    this.connect(outerPointList[i], innerPointList[scope[0]]);
                } else {
                    if (range[0] < range[1]) {
                        for (let j = range[0]; j <= range[1]; j++) {
                            this.connect(outerPointList[i], innerPointList[j]);
                        }
                    } else {
                        for (let j = range[0]; j <= range[1] + innerPointList.length; j++) {
                            this.connect(outerPointList[i], innerPointList[j % innerPointList.length]);
                        }
                    }
                    scope[0] = range[1];
                }
            }
        }
    }

    buildConnectionSchemeInternal(pointList) {
        const convexHullPointList = this.findConvexHull(pointList);
        const restPointList = pointList.filter(i => !convexHullPointList.includes(i));

        if (restPointList.length === 0) {
            this.connectInsideConvexHull(convexHullPointList);
        } else {
            const innerConvexHullPointList = this.buildConnectionSchemeInternal(restPointList);
            this.connectBetweenConvexHulls(convexHullPointList, innerConvexHullPointList);
        }

        return convexHullPointList;
    }

    buildConnectionScheme() {
        const pointList = [...this._points.keys()];
        const boundingPointList = this.buildConnectionSchemeInternal(pointList);
        return new Set(boundingPointList);
    }

    bestSchemeForAssemblyItem(pointList) {
        if (pointList.length === 3) {
            return [0.0, []];
        }

        let targetScheme = [Number.MAX_VALUE, []];

        if (this.anticlockwise(pointList[0], pointList[1], pointList[pointList.length - 1]) !== 0 &&
            this.anticlockwise(pointList[1], pointList[2], pointList[pointList.length - 1]) !== 0) {
            const scheme = this.bestSchemeForAssemblyItem(pointList.slice(1));
            scheme[0] += this.distance(pointList[1], pointList[pointList.length - 1]);
            scheme[1].push([pointList[1], pointList[pointList.length - 1]]);

            if (scheme[0] < targetScheme[0]) {
                targetScheme = scheme;
            }
        }
        for (let i = 2; i < pointList.length - 1; i++) {
            if (this.anticlockwise(pointList[0], pointList[1], pointList[i]) !== 0 &&
                this.anticlockwise(pointList[0], pointList[i], pointList[pointList.length - 1]) !== 0) {
                const schemel = this.bestSchemeForAssemblyItem(pointList.slice(0, i + 1));
                const schemer = this.bestSchemeForAssemblyItem([pointList[0], ...pointList.slice(i)]);
                const scheme = [
                    schemel[0] + this.distance(pointList[0], pointList[i]) + schemer[0],
                    [...schemel[1], [pointList[0], pointList[i]], ...schemer[1]]
                ];

                if (scheme[0] < targetScheme[0]) {
                    targetScheme = scheme;
                }
            }
        }

        return targetScheme;
    }

    currentSchemeForAssemblyItem(pointList) {
        const scheme = [0.0, []];

        for (let i = 0; i < pointList.length; i++) {
            const u = pointList[i];
            const v = pointList[(i + 1) % pointList.length];
            const w = pointList[(i - 1 + pointList.length) % pointList.length];

            for (const j of this._adjacencyVectorLink[u].aheadNodes(v)) {
                if (j === w) {
                    break;
                }

                if (u < j) {
                    scheme[0] += this.distance(u, j);
                    scheme[1].push([u, j]);
                }
            }
        }

        return scheme;
    }

    fineTuneAssemblyItem(pointList) {
        const currentScheme = this.currentSchemeForAssemblyItem(pointList);
        const bestScheme = this.bestSchemeForAssemblyItem(pointList);

        if (currentScheme[0] - bestScheme[0] > 1e-4) {
            for (const segment of currentScheme[1]) {
                this.disconnect(...segment);
            }
            for (const segment of bestScheme[1]) {
                this.connect(...segment);
            }
            return true;
        } else {
            return false;
        }
    }

    fineTune() {
        const log = [];

        let done = false;
        while (!done) {

            let items = [...this.triangles()];
            const available = new Set(items.map(i => i[0][0]));

            let turned = 0;
            for (let n = 4; turned === 0; n++) {
                items = [...this.polygons(items)];

                if (items.length === 0) {
                    done = true;
                    break;
                }

                for (const item of items) {
                    if (item[0].every(e => available.has(e))) {
                        if (this.fineTuneAssemblyItem(item[1])) {
                            for (const tid of item[0]) {
                                available.delete(tid);
                            }
                            turned += 1;
                        }
                    }
                }

                if (turned !== 0) {
                    log.push({ n, turned });
                }
            }
        }

        return log;
    }

    * triangles() {
        if (this._boundingPointList.size < 3) {
            return;
        }

        const boundingPointList = [...this._boundingPointList.values()];

        for (let i = 0; i < boundingPointList.length; i++) {
            const u = boundingPointList[i];
            const v = boundingPointList[(i + 1) % boundingPointList.length];

            let j = this._adjacencyVectorLink[u].allNodes().next().value;
            for (const k of this._adjacencyVectorLink[u].aheadNodes(j)) {
                if (k !== v && u < j && u < k) {
                    if (this.anticlockwise(u, j, k) !== 0) {
                        yield [[this.triangleId(u, j, k)], [u, j, k]];
                    }
                }
                j = k;
            }
        }

        for (let i = 0; i < this._points.length; i++) {
            if (this._boundingPointList.has(i)) {
                continue;
            }

            let j = this._adjacencyVectorLink[i].allNodes().next().value;
            for (const k of this._adjacencyVectorLink[i].aheadNodes(j)) {
                if (i < j && i < k) {
                    if (this.anticlockwise(i, j, k) !== 0) {
                        yield [[this.triangleId(i, j, k)], [i, j, k]];
                    }
                }
                j = k;
            }
        }
    }

    * polygons(prevLevelItems) {
        const existent = new Set();
        for (const [tids, pointList] of prevLevelItems) {
            for (let i = 0; i < pointList.length; i++) {
                const u = pointList[i];
                const v = pointList[(i + 1) % pointList.length];
                const w = this._adjacencyVectorLink[v].aheadNodes(u).next().value;
                if (this.anticlockwise(u, w, v) !== 0) {
                    const tid = this.triangleId(u, w, v);
                    if (tids[0] < tid) {
                        const k = [...pointList.slice(0, i + 1), w, ...pointList.slice(i + 1)];
                        if (this.convexHull(k)) {
                            const id = [...tids, tid].sort().join(',');
                            if (!existent.has(id)) {
                                yield [[...tids, tid], k];
                                existent.add(id);
                            }
                        }
                    }
                }
            }
        }
    }

    assemblies() {
        const list = [{ n: 3, items: [...this.triangles()] }];
        for (let n = 4; ; n++) {
            const items = [...this.polygons(list[list.length - 1].items)];
            if (items.length !== 0) {
                list.push({ n, items });
            } else {
                break;
            }
        }
        return list;
    }

    * segments() {
        const boundingPointList = [...this._boundingPointList.values()];

        if (this.collinear(boundingPointList)) { //All points in same line
            for (let i = 0; i < boundingPointList.length - 1; i++) {
                const u = boundingPointList[i];
                const v = boundingPointList[i + 1];

                if (u < v) {
                    yield [[u, v], true];
                } else {
                    yield [[v, u], true];
                }
            }
        } else {
            for (let i = 0; i < boundingPointList.length; i++) {
                const u = boundingPointList[i];
                const v = boundingPointList[(i + 1) % boundingPointList.length];
                const w = boundingPointList[(i - 1 + boundingPointList.length) % boundingPointList.length];

                if (u < v) {
                    yield [[u, v], true];
                } else {
                    yield [[v, u], true];
                }

                for (const j of this._adjacencyVectorLink[u].aheadNodes(v)) {
                    if (j === w) {
                        break;
                    }

                    if (u < j) {
                        yield [[u, j], false];
                    }
                }
            }

            for (let i = 0; i < this._points.length; i++) {
                if (this._boundingPointList.has(i)) {
                    continue;
                }
                for (const j of this._adjacencyVectorLink[i].allNodes()) {
                    if (i < j) {
                        yield [[i, j], false];
                    }
                }
            }
        }
    }
}

function generate(fileName, maxPointNumber, quantityForEachPointNumber) {
    fs.writeFileSync(fileName, '', { flag: 'w' });

    const randomEngine = new RandomEngine();

    for (let pointNumber = 5; pointNumber <= maxPointNumber; pointNumber++) {

        for (let i = 0; i < quantityForEachPointNumber; i++) {
            const points = [...randomPoints(randomEngine, pointNumber)];
            const scheme = new Scheme(points);
            scheme.fineTune();

            const segments = [...scheme.segments()].filter(i => !i[1]).map(i => i[0]);
            segments.sort((a, b) => a[0] !== b[0] ? a[0] - b[0] : a[1] - b[1]);

            const line = `${points.reduce((a, b) => a.concat(b.x, b.y), []).join(' ')} output ${segments.map(i => `${i[0] + 1} ${i[1] + 1}`).join(' ')} 0 0\n`;
            fs.writeFileSync(fileName, line, { flag: 'a' });
        }

        console.info(`Generating for point number (${pointNumber}) is done.`);
    }
}

generate('data/mwt_all_training.txt', 10, 100000);
generate('data/mwt_all_test.txt', 10, 1000);
