let currentInput = '';  
let lastOperation = ''; 
let previousInput = ''; 

function appendNumber(number) {
    currentInput += number;  
    updateDisplay();
}

function operation(op) {
    if (currentInput === '') return;  
    if (previousInput !== '') {
        calculateResult();  
    }
    lastOperation = op; 
    previousInput = currentInput;  
    currentInput = '';  
    updateDisplay();
}

function clearDisplay() {
    currentInput = '';  
    previousInput = '';
    lastOperation = '';
    updateDisplay();
}

function calculateResult() {
    if (previousInput === '' || currentInput === '') return;  
    let result;
    switch (lastOperation) {
        case '+':
            result = parseFloat(previousInput) + parseFloat(currentInput);
            break;
        case '-':
            result = parseFloat(previousInput) - parseFloat(currentInput);
            break;
        case '*':
            result = parseFloat(previousInput) * parseFloat(currentInput);
            break;
        case '/':
            result = parseFloat(previousInput) / parseFloat(currentInput);
            break;
        default:
            return;
    }
    currentInput = result.toString();  
    previousInput = '';  
    lastOperation = '';  
    updateDisplay();
}

function updateDisplay() {
    if (previousInput === '') {
        document.getElementById('display').value = currentInput;  
    } else {
        document.getElementById('display').value = previousInput + ' ' + lastOperation + ' ' + currentInput;  // Display the full expression (e.g., 2 + 3)
    }
}
