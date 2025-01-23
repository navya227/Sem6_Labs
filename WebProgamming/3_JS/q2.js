const greetingElement = document.getElementById("greeting");
const clockElement = document.getElementById("clock");

function updateClock() {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes();
    const seconds = now.getSeconds();

    const formattedTime = `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
    clockElement.textContent = formattedTime;
}

function setGreeting() {
    const now = new Date();
    const hours = now.getHours();

    if (hours >= 5 && hours < 12) {
        greetingElement.textContent = "Good Morning!";
        alert("It's Morning! Have a wonderful day !");
    } else if (hours >= 12 && hours < 17) {
        greetingElement.textContent = "Good Afternoon!";
        alert("It's Afternoon. You're halfway through, keep going");
    } else if (hours >= 17 && hours < 21) {
        greetingElement.textContent = "Good Evening!";
        alert("It's Evening! Almost done for the day");
    } else {
        greetingElement.textContent = "Good Night!";
        alert("It's Night! You can now relax");
    }
}

setGreeting();
updateClock();
setInterval(updateClock, 1000); 
