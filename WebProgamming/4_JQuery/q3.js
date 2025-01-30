$(document).ready(function() {
    function updateCard() {
        let bgColor = $("#bg-color").val();
        let font = $("#font-select").val();
        let fontSize = $("#font-size").val();
        let borderStyle = $("input[name='border-style']:checked").val();
        let addPicture = $("#add-picture").prop("checked");
        let greetingText = $("#greeting-text").val();

        let leftContainer = $(".form-container");
        let rightContainer = $(".card-container");

        leftContainer.css("background-color", bgColor);
        rightContainer.css("background-color", darkenColor(bgColor));

        $("body").css("font-family", font);
        $("#card-greeting").css({
            "font-family": font,
            "font-size": fontSize + "px"
        });

        if (borderStyle === "none") {
            rightContainer.css("border", "none");
        } else if (borderStyle === "single") {
            rightContainer.css("border", "1px solid #ddd");
        } else if (borderStyle === "double") {
            rightContainer.css("border", "4px double #ddd");
        }

        $("#card-greeting").text(greetingText);

        if (addPicture) {
            $("#card-image").attr("src", "bd.avif").show();
        } else {
            $("#card-image").hide();
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

    $("button").click(updateCard);  // Event listener for the button click
});
