const PolygonNames = {
    3: 'triangles',
    4: 'quadrilaterals',
    5: 'pentagons',
    6: 'hexagons',
    7: 'septagons',
    8: 'octagons',
    9: 'nonagons',
    10: 'decagons',
    11: 'undecagons',
    12: 'dodecagons',
    13: 'tridecagons',
    14: 'tetradecagons',
    15: 'pentedecagons',
    16: 'hexdecagons'
};

class RandomEngine {
    constructor() {
        this.size = 1024;
        this.array = new Uint32Array(this.size);
        this.refill();
    }

    refill() {
        self.crypto.getRandomValues(this.array);
        this.offset = 0;
    }

    next() {
        if (this.offset === this.size) {
            this.refill();
        }
        return this.array[this.offset++];
    }
}

function generatePoints(randomEngine, pointNumber) {
    const resolution = 1024;
    const grid = 16;
    const padding = 4;

    const points = [];

    const center = { x: resolution / 2, y: resolution / 2 };
    const radius2 = Math.pow(Math.min(resolution / 2, resolution / 2), 2);

    const pointList = new Set();
    while (points.length < pointNumber) {
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

        points.push({ x: x / resolution, y: y / resolution });
        pointList.add(pointIndex);
    }

    return points;
}

class App extends React.Component {
    constructor(props) {
        super(props);
        this.randomEngine = new RandomEngine();
        this.state = {
            pointNumber: 9,
            points: [],
            boundingLineList: [],
            internalLineNumber: 0,
            schemes: [],
            activeSchemeIndex: -1,
            activeAssemblyIndex: -1,
            activeAssemblyItemIndex: -1,
            showPoint: true,
            showPointLabel: true,
            showEdgeLabel: false
        };
        this.handlePointNumberChange = this.handlePointNumberChange.bind(this);
        this.handleRandomGeneratePoints = this.handleRandomGeneratePoints.bind(this);
        this.handleImportPoints = this.handleImportPoints.bind(this);
        this.handleExportPoints = this.handleExportPoints.bind(this);
        this.handleShowPointChange = this.handleShowPointChange.bind(this);
        this.handleShowPointLabelChange = this.handleShowPointLabelChange.bind(this);
        this.handleShowEdgeLabelChange = this.handleShowEdgeLabelChange.bind(this);
        this.handleFineTune = this.handleFineTune.bind(this);
        this.handleSchemeShow = this.handleSchemeShow.bind(this);
        this.handleShowAssembly = this.handleShowAssembly.bind(this);
        this.handleAssemblyItemPrev = this.handleAssemblyItemPrev.bind(this);
        this.handleAssemblyItemNext = this.handleAssemblyItemNext.bind(this);
    }

    initialize(points, boundingLineList, internalLineNumber, callback) {
        this.setState({ points, boundingLineList, internalLineNumber, schemes: [], activeSchemeIndex: -1, activeAssemblyIndex: -1 }, callback);
    }

    addScheme(internalLineList, internalLineTotalLength, assemblies) {
        this.setState(state => {
            const scheme = {
                internalLineList,
                internalLineTotalLength,
                assemblies
            };
            state.schemes.push(scheme);
            state.activeSchemeIndex = state.schemes.length - 1;
            return state;
        });
    }

    createNewScheme(points) {
        console.clear();
        this.scheme = new Scheme(points);
        const segments = [...this.scheme.segments()];
        const boundingSegments = segments.filter(([_, t]) => t).map(([s, _]) => s);
        const internalSegments = segments.filter(([_, t]) => !t).map(([s, _]) => s);
        const internalSegmentTotalLength = internalSegments.reduce((length, [i, j]) => length + distance(points[i], points[j]), 0.0);
        const assemblies = this.scheme.assemblies();
        this.initialize(
            points,
            boundingSegments,
            internalSegments.length,
            () => this.addScheme(internalSegments, internalSegmentTotalLength, assemblies));
    }

    handlePointNumberChange(e) {
        if (e.target.value >= 0 && e.target.value <= 1024) {
            this.setState({
                pointNumber: e.target.value
            });
        }
    }

    handleRandomGeneratePoints() {
        this.createNewScheme(generatePoints(this.randomEngine, this.state.pointNumber));
    }

    handleImportPoints() {
        const pointText = prompt('Points JSON string:');
        this.createNewScheme(JSON.parse(pointText));
    }

    handleExportPoints() {
        const pointText = JSON.stringify(this.state.points);
        navigator.clipboard.writeText(pointText).then(() => alert('Export points data to clipboard successfully!'));
    }

    handleShowPointChange(e) {
        this.setState({ showPoint: e.target.checked });
    }

    handleShowPointLabelChange(e) {
        this.setState({ showPointLabel: e.target.checked });
    }

    handleShowEdgeLabelChange(e) {
        this.setState({ showEdgeLabel: e.target.checked });
    }

    handleFineTune() {
        const logItems = this.scheme.fineTune();
        const segments = [...this.scheme.segments()];
        const internalSegments = segments.filter(([_, t]) => !t).map(([s, _]) => s);
        const internalSegmentTotalLength = internalSegments.reduce((length, [i, j]) => length + distance(this.state.points[i], this.state.points[j]), 0.0);
        const assemblies = this.scheme.assemblies();
        const messages = logItems.map((item, index) => `Round ${index + 1}, turned ${item.turned} ${PolygonNames[item.n]}`);
        this.addScheme(internalSegments, internalSegmentTotalLength, assemblies, messages);
    }

    handleSchemeShow(e) {
        const index = parseInt(e.currentTarget.dataset.scheme_index);
        this.setState({ activeSchemeIndex: index });
    }

    handleShowAssembly(e) {
        e.stopPropagation();
        const schemeIndex = parseInt(e.target.dataset.scheme_index);
        const assemblyIndex = parseInt(e.target.dataset.assembly_index)
        this.setState({
            activeSchemeIndex: schemeIndex,
            activeAssemblyIndex: assemblyIndex,
            activeAssemblyItemIndex: 0
        });
    }

    handleAssemblyItemPrev(e) {
        this.setState(state => {
            const activeAssembly = state.schemes[state.activeSchemeIndex].assemblies[state.activeAssemblyIndex];
            state.activeAssemblyItemIndex = (state.activeAssemblyItemIndex - 1 + activeAssembly.items.length) % activeAssembly.items.length;
            return state;
        });
    }

    handleAssemblyItemNext(e) {
        this.setState(state => {
            const activeAssembly = state.schemes[state.activeSchemeIndex].assemblies[state.activeAssemblyIndex];
            state.activeAssemblyItemIndex = (state.activeAssemblyItemIndex + 1) % activeAssembly.items.length;
            return state;
        });
    }

    componentDidMount() {
        this.handleRandomGeneratePoints();
    }

    renderBoundingLines() {
        return this.state.boundingLineList.map(([index1, index2], i) => {
            const p1 = this.state.points[index1];
            const p2 = this.state.points[index2];
            return <React.Fragment key={i}>
                <line
                    x1={p1.x * this.props.width}
                    y1={p1.y * this.props.height}
                    x2={p2.x * this.props.width}
                    y2={p2.y * this.props.height} stroke="black" strokeWidth="1.5" />
                {this.state.showEdgeLabel && <text
                    x={(p1.x + p2.x) / 2 * this.props.width}
                    y={(p1.y + p2.y) / 2 * this.props.height} stroke="blue">{`${index1 + 1}-${index2 + 1}`}</text>}
            </React.Fragment>;
        });
    }

    renderInternalLines() {
        const internalLineList = this.state.activeSchemeIndex !== -1 ? this.state.schemes[this.state.activeSchemeIndex].internalLineList : [];
        return internalLineList.map(([index1, index2], i) => {
            const p1 = this.state.points[index1];
            const p2 = this.state.points[index2];
            return <React.Fragment key={i}>
                <line x1={p1.x * this.props.width}
                    y1={p1.y * this.props.height}
                    x2={p2.x * this.props.width}
                    y2={p2.y * this.props.height} stroke="silver" strokeWidth="1.5" />
                {this.state.showEdgeLabel && <text
                    x={(p1.x + p2.x) / 2 * this.props.width}
                    y={(p1.y + p2.y) / 2 * this.props.height} stroke="blue">{`${index1 + 1}-${index2 + 1}`}</text>}
            </React.Fragment>;
        });
    }

    renderPoints() {
        if (this.state.showPoint) {
            return this.state.points.map((point, pointIndex) => <React.Fragment key={pointIndex}>
                <circle cx={point.x * this.props.width} cy={point.y * this.props.height} r="3" fill="red" />
                {this.state.showPointLabel && <text x={point.x * this.props.width} y={point.y * this.props.height} stroke="brown">{pointIndex + 1}</text>}
            </React.Fragment>);
        }
    }

    renderAssemblyItem() {
        if (this.state.activeSchemeIndex !== -1 && this.state.activeAssemblyIndex !== -1) {
            const activeScheme = this.state.schemes[this.state.activeSchemeIndex];
            const activeAssembly = activeScheme.assemblies[this.state.activeAssemblyIndex];
            const activeAssemblyItem = activeAssembly.items[this.state.activeAssemblyItemIndex];
            const pointList = activeAssemblyItem[1];
            const pointText = pointList.map(i => this.state.points[i]).map(p => `${p.x * this.props.width},${p.y * this.props.height}`).join(' ');
            return <React.Fragment>
                <polygon points={pointText} fill="lightyellow" />
            </React.Fragment>;
        }
    }

    renderAssemblyPanel() {
        if (this.state.activeSchemeIndex !== -1 && this.state.activeAssemblyIndex !== -1) {
            const activeScheme = this.state.schemes[this.state.activeSchemeIndex];
            const activeAssembly = activeScheme.assemblies[this.state.activeAssemblyIndex];
            const activeAssemblyItem = activeAssembly.items[this.state.activeAssemblyItemIndex];
            return <div>
                <span>Showing {PolygonNames[activeAssembly.n]}&nbsp;[</span>
                <span>bounding points:&nbsp;<b>{activeAssemblyItem[1].map(i => i + 1).join('-')}</b></span>
                <span>]&nbsp;({this.state.activeAssemblyItemIndex + 1}/{activeAssembly.items.length})&nbsp;</span>
                <button onClick={this.handleAssemblyItemPrev}>&lt;</button>
                <span>&nbsp;</span>
                <button onClick={this.handleAssemblyItemNext}>&gt;</button>
            </div>;
        }
    }


    render() {
        return <React.Fragment>
            <div>
                <span>Triangulation for <input type="number" value={this.state.pointNumber} onChange={this.handlePointNumberChange} /> vertices&nbsp;&nbsp;</span>
                <button onClick={this.handleRandomGeneratePoints}>Random generate</button>
                <span>&nbsp;&nbsp;</span>
                <button onClick={this.handleImportPoints}>Import</button>
                <span>&nbsp;&nbsp;</span>
                <button onClick={this.handleExportPoints}>Export</button>
                <span>&nbsp;&nbsp;</span>
                <span><button onClick={this.handleFineTune}>Fine tune</button></span>
            </div>
            <div className="container">
                <div>
                    <svg width={this.props.width} height={this.props.height} xmlns="http://www.w3.org/2000/svg">
                        <g>
                            {this.renderAssemblyItem()}
                            {this.renderInternalLines()}
                            {this.renderBoundingLines()}
                            {this.renderPoints()}
                        </g>
                    </svg>
                    {this.renderAssemblyPanel()}
                </div>
                <div>
                    <div className="legend">
                        <b>Parameters:</b>
                        <p>Vertex number: {this.state.points.length}</p>
                        <p>Edge number: {this.state.boundingLineList.length + this.state.internalLineNumber}</p>
                        <p>Triangle number: {(this.state.boundingLineList.length + this.state.internalLineNumber * 2) / 3}</p>
                        <hr />
                        <b>Options:</b>
                        <p>Show points: <input type="checkbox" checked={this.state.showPoint} onChange={this.handleShowPointChange} /></p>
                        <p>Show point labels: <input type="checkbox" checked={this.state.showPointLabel} onChange={this.handleShowPointLabelChange} /></p>
                        <p>Show edge labels: <input type="checkbox" checked={this.state.showEdgeLabel} onChange={this.handleShowEdgeLabelChange} /></p>
                        <hr />
                        <b>Schemes:</b>
                        {this.state.schemes.map((scheme, schemeIndex) => <div key={schemeIndex} data-scheme_index={schemeIndex} className={`card ${this.state.activeSchemeIndex === schemeIndex && 'activecard'}`} onClick={this.handleSchemeShow}>
                            <p>Total length: {scheme.internalLineTotalLength.toFixed(4)}</p>
                            <p>{scheme.assemblies.map((assembly, assemblyIndex) => assembly.items.length !== 0 && <span key={assemblyIndex}>
                                <a href="#" onClick={this.handleShowAssembly} data-scheme_index={schemeIndex} data-assembly_index={assemblyIndex}>{PolygonNames[assembly.n]}</a>
                                <span>&nbsp;({assembly.items.length})&nbsp;</span>
                            </span>)}</p>
                        </div>)}
                    </div>
                </div>
            </div>
        </React.Fragment>;
    }
}

ReactDOM.render(<App width={960} height={960} />, document.querySelector('#root'));
