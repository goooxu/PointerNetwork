
class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            points: [],
            P_boundingLineList: [],
            Y_boundingLineList: [],
            showPoint: true,
            showPointLabel: true
        };
        this.handleInputTextChange = this.handleInputTextChange.bind(this);
        this.handleShowPointChange = this.handleShowPointChange.bind(this);
        this.handleShowPointLabelChange = this.handleShowPointLabelChange.bind(this);
    }

    handleInputTextChange(e) {
        const inputText = e.target.value;
        const inputObject = JSON.parse(inputText);

        console.log(inputObject);

        const points = [];
        for (let i = 0; i < inputObject[0].length; i++) {
            points.push({
                x: inputObject[0][i][0],
                y: inputObject[0][i][1]
            });
        }
        const Y_boundingLineList = [];
        for (let i = 0; i < inputObject[1].length - 1; i++) {
            Y_boundingLineList.push([
                inputObject[1][i] - 1,
                inputObject[1][i + 1] - 1
            ]);
        }
        const P_boundingLineList = [];
        for (let i = 0; i < inputObject[2].length - 1; i++) {
            P_boundingLineList.push([
                inputObject[2][i] - 1,
                inputObject[2][i + 1] - 1
            ]);
        }

        this.setState({ points, Y_boundingLineList, P_boundingLineList });
    }

    handleShowPointChange(e) {
        this.setState({ showPoint: e.target.checked });
    }

    handleShowPointLabelChange(e) {
        this.setState({ showPointLabel: e.target.checked });
    }

    componentDidMount() {
    }

    renderBoundingLines(lineList, color) {
        return lineList.map((line, index) => <React.Fragment key={index}>
            <line
                x1={this.state.points[line[0]].x * this.props.width}
                y1={this.props.height - this.state.points[line[0]].y * this.props.height}
                x2={this.state.points[line[1]].x * this.props.width}
                y2={this.props.height - this.state.points[line[1]].y * this.props.height}
                stroke={color} strokeWidth="2.5" />
        </React.Fragment>);
    }

    renderPoints() {
        return this.state.points.map((point, pointIndex) => <React.Fragment key={pointIndex}>
            <circle cx={point.x * this.props.width} cy={this.props.height - point.y * this.props.height} r="3" fill="black" />
            {this.state.showPointLabel && <text x={point.x * this.props.width + 2} y={this.props.height - point.y * this.props.height - 2} stroke="royalblue">{pointIndex + 1}</text>}
        </React.Fragment>);
    }

    render() {
        return <React.Fragment>
            <div className="container">
                <div>
                    <textarea value={this.state.inputText} onChange={this.handleInputTextChange} />
                    <svg width={this.props.width} height={this.props.height} xmlns="http://www.w3.org/2000/svg">
                        <g>
                            {this.renderBoundingLines(this.state.P_boundingLineList, "red")}
                            {this.renderBoundingLines(this.state.Y_boundingLineList, "lime")}
                            {this.state.showPoint && this.renderPoints()};
                    </g>
                    </svg>
                </div>

                <div>
                    <div className="legend">
                        <b>Parameters:</b>
                        <p>Vertex number: {this.state.points.length}</p>
                        <b>Options:</b>
                        <p>Show points: <input type="checkbox" checked={this.state.showPoint} onChange={this.handleShowPointChange} /></p>
                        <p>Show point labels: <input type="checkbox" checked={this.state.showPointLabel} onChange={this.handleShowPointLabelChange} /></p>
                    </div>
                </div>
            </div>
        </React.Fragment>;
    }
}

ReactDOM.render(<App width={800} height={800} />, document.querySelector('#root'));