function calculateGrade() {
    const subject1 = parseFloat(document.getElementById("subject1").value);
    const subject2 = parseFloat(document.getElementById("subject2").value);
    const subject3 = parseFloat(document.getElementById("subject3").value);
    const subject4 = parseFloat(document.getElementById("subject4").value);

    if (isNaN(subject1) || isNaN(subject2) || isNaN(subject3) || isNaN(subject4)) {
        document.getElementById("result").textContent = "Please enter valid marks for all subjects.";
        return;
    }

    const average = (subject1 + subject2 + subject3 + subject4) / 4;

    let grade;
    if (average > 90) {
        grade = "A";
    } else if (average > 80) {
        grade = "B";
    } else if (average > 70) {
        grade = "C";
    } else if (average > 60) {
        grade = "D";
    } else {
        grade = "F";
    }

    document.getElementById("result").textContent = `Average: ${average.toFixed(2)}, Grade: ${grade}`;
}

document.getElementById("calculateBtn").addEventListener("click", calculateGrade);
