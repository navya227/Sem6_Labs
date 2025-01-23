const questions = [
    {
        question: "Who is the father of Computers?",
        options: ["James Gosling", "Charles Babbage", "Dennis Ritchie", "Bjarne Stroustrup"],
        answer: 1, 
        type: "multiple-choice", 
    },
    {
        question: "What is the full form of CPU?",
        options: ["Computer Processing Unit", "Computer Principle Unit", "Central Processing Unit", "Control Processing Unit"],
        answer: 2,
        type: "multiple-choice",
    },
    {
        question: "Which of the following computer language is written in binary codes only?",
        options: ["Pascal", "Machine Language", "C", "C#"],
        answer: 1,
        type: "multiple-choice",
    },
    {
        question: "What is 2 + 2?", 
        options: [], 
        answer: "4", 
        type: "text",
    },
];


let currentQuestionIndex = 0;
let score = 0;

const questionElement = document.getElementById("question");
const optionsElement = document.getElementById("options");
const nextButton = document.getElementById("next-btn");
const resultElement = document.getElementById("result");

function showQuestion() {
    const currentQuestion = questions[currentQuestionIndex];
    questionElement.textContent = currentQuestion.question;
    optionsElement.innerHTML = "";

    if (currentQuestion.type === "multiple-choice") {
        currentQuestion.options.forEach((option, index) => {
            const li = document.createElement("li");
            const label = document.createElement("label");
            const radio = document.createElement("input");

            radio.type = "radio";
            radio.name = "option";
            radio.value = index;

            label.appendChild(radio);
            label.appendChild(document.createTextNode(option));
            li.appendChild(label);
            optionsElement.appendChild(li);
        });
    } else if (currentQuestion.type === "text") {
        const input = document.createElement("input");
        input.type = "text";
        input.id = "text-answer";
        input.placeholder = "Enter your answer here";
        optionsElement.appendChild(input);
    }
}

function getSelectedOption() {
    const currentQuestion = questions[currentQuestionIndex];

    if (currentQuestion.type === "multiple-choice") {
        const selectedOption = document.querySelector('input[name="option"]:checked');
        return selectedOption ? parseInt(selectedOption.value) : null;
    } else if (currentQuestion.type === "text") {
        const textAnswer = document.getElementById("text-answer").value.trim();
        return textAnswer ? textAnswer : null;
    }
    return null;
}

nextButton.addEventListener("click", () => {
    const currentQuestion = questions[currentQuestionIndex];
    const selectedOption = getSelectedOption();

    if (selectedOption === null) {
        alert("Please provide an answer before proceeding.");
        return;
    }

    if (currentQuestion.type === "multiple-choice") {
        if (selectedOption === currentQuestion.answer) {
            score++;
        }
    } else if (currentQuestion.type === "text") {
        if (selectedOption.toLowerCase() === currentQuestion.answer.toLowerCase()) {
            score++;
        }
    }

    currentQuestionIndex++;

    if (currentQuestionIndex < questions.length) {
        showQuestion();
    } else {
        showResult();
    }
});

function showResult() {
    questionElement.style.display = "none";
    optionsElement.style.display = "none";
    nextButton.style.display = "none";
    resultElement.textContent = `Your score is ${score} out of ${questions.length}`;
}

// Start the quiz
showQuestion();
