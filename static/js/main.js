const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileNameDisplay = document.getElementById("fileNameDisplay");
const extractBtn = document.getElementById("extractBtn");

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        fileNameDisplay.innerText = "Selected: " + fileInput.files[0].name;
    }
});

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "#00c6ff";
});

dropZone.addEventListener("dragleave", () => {
    dropZone.style.borderColor = "#ccc";
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "#ccc";
    fileInput.files = e.dataTransfer.files;
    if (fileInput.files.length > 0) {
        fileNameDisplay.innerText = "Dropped: " + fileInput.files[0].name;
    }
});

async function extractData() {
    const file = fileInput.files[0];
    if (!file) {
        alert("رجاءً اختر ملف أولاً!");
        return;
    }

    extractBtn.innerText = "Processing... Please wait";
    extractBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/api/v1/extract", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.status === "success") {
            displayResults(result.data);
        } else {
            alert("Error: " + (result.detail || "Something went wrong"));
        }
    } catch (error) {
        console.error("Fetch Error:", error);
        alert("فشل الاتصال بالسيرفر. تأكد أن السيرفر يعمل.");
    } finally {
        extractBtn.innerText = "Extract Information";
        extractBtn.disabled = false;
    }
}

function displayResults(data) {
    document.getElementById("name").textContent = data.name || "Not found";
    document.getElementById("email").textContent = data.email || "Not found";
    
    document.getElementById("skills").textContent = 
        (data.skills && data.skills.length > 0) ? data.skills.join(", ") : "Not found";
    
    document.getElementById("education").textContent = 
        (data.education && data.education.length > 0) ? data.education.join("; ") : "Not found";

}