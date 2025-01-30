$(document).ready(function() {
    const productPrices = {
        HP: { mobile: 500, laptop: 1000 },
        Nokia: { mobile: 200, laptop: 800 },
        Samsung: { mobile: 400, laptop: 900 },
        Motorola: { mobile: 300, laptop: 850 },
        Apple: { mobile: 1000, laptop: 1500 }
    };

    $("#produceBillButton").click(function() {
        const brand = $("#brand").val();
        const mobileChecked = $("#mobile").prop("checked");
        const laptopChecked = $("#laptop").prop("checked");
        const quantity = parseInt($("#quantity").val());

        if (!quantity || (!mobileChecked && !laptopChecked)) {
            alert("Please select items and enter a valid quantity.");
            return;
        }

        let totalAmount = 0;

        if (mobileChecked) {
            totalAmount += productPrices[brand].mobile * quantity;
        }

        if (laptopChecked) {
            totalAmount += productPrices[brand].laptop * quantity;
        }

        alert(`Total Amount: $${totalAmount}`);
    });
});
