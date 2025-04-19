// DOM Elements
const runButton = document.getElementById('run-button');
const preset1Button = document.getElementById('preset-1');
const preset2Button = document.getElementById('preset-2');
const preset3Button = document.getElementById('preset-3');
const stepTable = document.getElementById('step-table').getElementsByTagName('tbody')[0];
const currentStepInfo = document.getElementById('current-step-info');
const errorMessage = document.getElementById('error-message');

// Store current state
let currentHistory = [];
let currentStep = 0;
let surfacePlot = null;
let contourPlot = null;

// Get parameters from inputs
function getParameters() {
    return {
        a: parseFloat(document.getElementById('param-a').value),
        b: parseFloat(document.getElementById('param-b').value),
        c: parseFloat(document.getElementById('param-c').value),
        d: parseFloat(document.getElementById('param-d').value),
        e: parseFloat(document.getElementById('param-e').value),
        f: parseFloat(document.getElementById('param-f').value),
        start_w1: parseFloat(document.getElementById('start-w1').value),
        start_w2: parseFloat(document.getElementById('start-w2').value),
        learning_rate: parseFloat(document.getElementById('learning-rate').value),
        max_steps: parseInt(document.getElementById('max-steps').value)
    };
}

// Set parameters to inputs
function setParameters(params) {
    document.getElementById('param-a').value = params.a;
    document.getElementById('param-b').value = params.b;
    document.getElementById('param-c').value = params.c;
    document.getElementById('param-d').value = params.d;
    document.getElementById('param-e').value = params.e;
    document.getElementById('param-f').value = params.f;
    document.getElementById('start-w1').value = params.start_w1;
    document.getElementById('start-w2').value = params.start_w2;
    document.getElementById('learning-rate').value = params.learning_rate;
    document.getElementById('max-steps').value = params.max_steps;
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Initialize 3D surface plot
function initSurfacePlot() {
    const layout = {
        title: 'Function Surface',
        autosize: true,
        margin: { l: 0, r: 0, b: 0, t: 30 }
    };
    
    Plotly.newPlot('surface-plot', [], layout);
}

// Initialize contour plot
function initContourPlot() {
    const layout = {
        title: 'Contour Plot with Gradient Descent Path',
        xaxis: { title: 'w₁' },
        yaxis: { title: 'w₂' },
        autosize: true,
        margin: { l: 50, r: 0, b: 50, t: 30 }
    };
    
    Plotly.newPlot('contour-plot', [], layout);
}

// Update surface plot
function updateSurfacePlot(surfaceData, history) {
    if (!surfaceData || !history) {
        console.error('Invalid data for surface plot');
        showError('Error: Invalid data for surface plot');
        return;
    }
    
    console.log('Surface data length:', surfaceData.length);
    console.log('History length:', history.length);
    
    try {
        const w1 = surfaceData.map(point => point.w1);
        const w2 = surfaceData.map(point => point.w2);
        const z = surfaceData.map(point => point.z);
        
        if (w1.length === 0 || w2.length === 0 || z.length === 0) {
            console.error('Empty data arrays for surface plot');
            showError('Error: Empty data for surface plot');
            return;
        }
        
        // Check if arrays have unique values for grid representation
        const uniqueW1 = [...new Set(w1)].length;
        const uniqueW2 = [...new Set(w2)].length;
        
        if (uniqueW1 * uniqueW2 !== w1.length) {
            console.error('Surface data does not form a proper grid');
            showError('Error: Surface data is not properly structured');
            return;
        }
        
        // Path coordinates
        const pathW1 = history.map(point => point.w1);
        const pathW2 = history.map(point => point.w2);
        const pathZ = history.map(point => point.f);
        
        // Create data with structured arrays 
        const data = [
            {
                type: 'surface',
                x: [...new Set(w1)],
                y: [...new Set(w2)],
                z: reshapeToGrid(w1, w2, z),
                colorscale: 'Viridis',
                opacity: 0.8
            },
            {
                type: 'scatter3d',
                x: pathW1,
                y: pathW2,
                z: pathZ,
                mode: 'lines+markers',
                line: { color: 'red', width: 6 },
                marker: { size: 5, color: 'red' }
            }
        ];
        
        Plotly.react('surface-plot', data);
    } catch (error) {
        console.error('Error updating surface plot:', error);
        showError('Error rendering surface plot: ' + error.message);
    }
}

// Helper function to reshape flat data to grid format
function reshapeToGrid(x, y, z) {
    // Get unique x and y values
    const uniqueX = [...new Set(x)].sort((a, b) => a - b);
    const uniqueY = [...new Set(y)].sort((a, b) => a - b);
    
    // Create empty 2D array
    const grid = Array(uniqueY.length).fill().map(() => Array(uniqueX.length).fill(null));
    
    // Fill the grid
    for (let i = 0; i < x.length; i++) {
        const xIndex = uniqueX.indexOf(x[i]);
        const yIndex = uniqueY.indexOf(y[i]);
        if (xIndex >= 0 && yIndex >= 0) {
            grid[yIndex][xIndex] = z[i];
        }
    }
    
    return grid;
}

// Update contour plot
function updateContourPlot(contourData, history) {
    if (!contourData || !history || !contourData.w1_values || !contourData.w2_values || !contourData.z_values) {
        console.error('Invalid data for contour plot');
        return;
    }
    
    console.log('Contour data dimensions:', contourData.z_values.length, 'x', 
                contourData.z_values[0]?.length);
    
    const { w1_values, w2_values, z_values, levels } = contourData;
    
    // Path coordinates
    const pathW1 = history.map(point => point.w1);
    const pathW2 = history.map(point => point.w2);
    
    const data = [
        {
            type: 'contour',
            z: z_values,
            x: w1_values,
            y: w2_values,
            colorscale: 'Viridis',
            contours: {
                coloring: 'heatmap',
                showlabels: true
            }
        },
        {
            type: 'scatter',
            x: pathW1,
            y: pathW2,
            mode: 'lines+markers',
            line: { color: 'red', width: 3 },
            marker: { size: 8, color: 'red' }
        }
    ];
    
    try {
        Plotly.react('contour-plot', data);
    } catch (error) {
        console.error('Error updating contour plot:', error);
        showError('Error rendering contour plot. Check console for details.');
    }
}

// Update step table
function updateStepTable(history) {
    // Clear table
    stepTable.innerHTML = '';
    
    // Add new rows
    history.forEach((step, index) => {
        const row = stepTable.insertRow();
        
        // Add cells
        row.insertCell(0).textContent = step.step;
        row.insertCell(1).textContent = step.w1.toFixed(4);
        row.insertCell(2).textContent = step.w2.toFixed(4);
        row.insertCell(3).textContent = step.f.toFixed(4);
        row.insertCell(4).textContent = step.grad_w1.toFixed(4);
        row.insertCell(5).textContent = step.grad_w2.toFixed(4);
        
        // Highlight current step
        if (index === currentStep) {
            row.style.backgroundColor = '#e6f7ff';
        }
        
        // Add click event
        row.style.cursor = 'pointer';
        row.onclick = () => {
            currentStep = index;
            updateStepTable(history);
            updateCurrentStepInfo(history[currentStep]);
        };
    });
}

// Update current step info
function updateCurrentStepInfo(step) {
    if (!step) return;
    
    let nextStep = null;
    if (currentStep < currentHistory.length - 1) {
        nextStep = currentHistory[currentStep + 1];
    }
    
    let html = `
        <p><strong>Step ${step.step}:</strong></p>
        <p>Current position: w₁ = ${step.w1.toFixed(4)}, w₂ = ${step.w2.toFixed(4)}</p>
        <p>Function value: f(w₁,w₂) = ${step.f.toFixed(4)}</p>
        <p>Gradient: [∂f/∂w₁ = ${step.grad_w1.toFixed(4)}, ∂f/∂w₂ = ${step.grad_w2.toFixed(4)}]</p>
    `;
    
    if (nextStep) {
        const lr = parseFloat(document.getElementById('learning-rate').value);
        html += `
            <p>Update:</p>
            <p>w₁_new = ${step.w1.toFixed(4)} - ${lr} × ${step.grad_w1.toFixed(4)} = ${nextStep.w1.toFixed(4)}</p>
            <p>w₂_new = ${step.w2.toFixed(4)} - ${lr} × ${step.grad_w2.toFixed(4)} = ${nextStep.w2.toFixed(4)}</p>
        `;
    }
    
    currentStepInfo.innerHTML = html;
}

// Run gradient descent
async function runGradientDescent() {
    hideError();
    const params = getParameters();
    try {
        const response = await fetch('/run_gradient_descent', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        });
        const data = await response.json();
        if (!data.success) {
            showError(data.error || 'Unknown error occurred');
            return;
        }
        
        // Store history
        currentHistory = data.history;
        currentStep = 0;
        
        // Update visualizations
        updateSurfacePlot(data.surface_data, data.history);
        updateContourPlot(data.contour_data, data.history);
        updateStepTable(data.history);
        updateCurrentStepInfo(data.history[currentStep]);
        
    } catch (error) {
        console.error('Error running gradient descent:', error);
        showError('Error running gradient descent. See console for details.');
    }
}

// Preset 1: Convex Function
function setPreset1() {
    const params = {
        a: 1,
        b: 1,
        c: 0,
        d: 0,
        e: 0,
        f: 0,
        start_w1: 2,
        start_w2: 2,
        learning_rate: 0.1,
        max_steps: 20
    };
    
    setParameters(params);
    runGradientDescent();
}

// Preset 2: Saddle Point
function setPreset2() {
    const params = {
        a: -1,
        b: 1,
        c: 0,
        d: 0,
        e: 0,
        f: 0,
        start_w1: 0.5,
        start_w2: 2,
        learning_rate: 0.1,
        max_steps: 20
    };
    
    setParameters(params);
    runGradientDescent();
}

// Preset 3: Shifted Minimum
function setPreset3() {
    const params = {
        a: 1,
        b: 1,
        c: 0,
        d: -2,
        e: -4,
        f: 0,
        start_w1: -1,
        start_w2: -1,
        learning_rate: 0.1,
        max_steps: 20
    };
    
    setParameters(params);
    runGradientDescent();
}

function initialize() {
    // Initialize plots
    initSurfacePlot();
    initContourPlot();
    
    // Add event listeners
    runButton.addEventListener('click', runGradientDescent);
    preset1Button.addEventListener('click', setPreset1);
    preset2Button.addEventListener('click', setPreset2);
    preset3Button.addEventListener('click', setPreset3);
    
    // Run initial visualization
    setPreset1();
}

// Load plots when document is ready
document.addEventListener('DOMContentLoaded', initialize);