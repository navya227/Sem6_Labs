const productPrices = {
    HP: { mobile: 500, laptop: 1000 },
    Nokia: { mobile: 200, laptop: 800 },
    Samsung: { mobile: 400, laptop: 900 },
    Motorola: { mobile: 300, laptop: 850 },
    Apple: { mobile: 1000, laptop: 1500 }
};

function produceBill() {
    const brand = document.getElementById("brand").value;
    const mobileChecked = document.getElementById("mobile").checked;
    const laptopChecked = document.getElementById("laptop").checked;
    const quantity = parseInt(document.getElementById("quantity").value);

    if (!quantity || (mobileChecked === false && laptopChecked === false)) {
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
}
