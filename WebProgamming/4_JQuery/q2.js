$(document).ready(function() {
    let currentInput = '';  
    let lastOperation = ''; 
    let previousInput = ''; 

    function updateDisplay() {
        if (previousInput === '') {
            $('#display').val(currentInput);  
        } else {
            $('#display').val(previousInput + ' ' + lastOperation + ' ' + currentInput);  
        }
    }

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

    $('.buttons button').click(function() {
        let value = $(this).text();  
        
        if ($.isNumeric(value)) {
            appendNumber(value);
        } else if (value === 'C') {
            clearDisplay();
        } else if (value === '=') {
            calculateResult();
        } else {
            operation(value);
        }
    });
});
