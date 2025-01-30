function updateCard() {

    let bgColor = document.getElementById("bg-color").value;
    let font = document.getElementById("font-select").value;
    let fontSize = document.getElementById("font-size").value;
    let borderStyle = document.querySelector('input[name="border-style"]:checked').value;
    let addPicture = document.getElementById("add-picture").checked;
    let greetingText = document.getElementById("greeting-text").value;
    

    let leftContainer = document.querySelector(".form-container");
    let rightContainer = document.querySelector(".card-container");
    
    leftContainer.style.backgroundColor = bgColor;
    rightContainer.style.backgroundColor = darkenColor(bgColor);

    document.body.style.fontFamily = font;
    document.getElementById("card-greeting").style.fontFamily = font;
    document.getElementById("card-greeting").style.fontSize = fontSize + "px";

    if (borderStyle === "none") {
        rightContainer.style.border = "none";
    } else if (borderStyle === "single") {
        rightContainer.style.border = "1px solid #ddd";
    } else if (borderStyle === "double") {
        rightContainer.style.border = "4px double #ddd";
    }


    document.getElementById("card-greeting").textContent = greetingText;


    if (addPicture) {
        document.getElementById("card-image").src = "bd.avif";
        document.getElementById("card-image").style.display = "block";
    } else {
        document.getElementById("card-image").style.display = "none";
    }
}


function darkenColor(color) {
    let colors = {
        yellow: "#FFB200",
        lightblue: "#4682B4",
        lightgreen: "#2E8B57",
        lightpink: "#d66fa8"
    };
    return colors[color] || "#000000";
}
