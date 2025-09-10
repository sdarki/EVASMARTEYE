// This file contains JavaScript code for client-side functionality.
// You can add scripts for form validation, dynamic content updates, or handling user interactions here.

document.addEventListener("DOMContentLoaded", function() {
    // Example: Form validation for the login form
    const loginForm = document.getElementById("loginForm");
    if (loginForm) {
        loginForm.addEventListener("submit", function(event) {
            const username = document.getElementById("username").value;
            const password = document.getElementById("password").value;
            if (!username || !password) {
                event.preventDefault();
                alert("Please fill in both username and password.");
            }
        });
    }

    // Example: Confirmation before deleting a camera
    const deleteButtons = document.querySelectorAll(".delete-camera");
    deleteButtons.forEach(button => {
        button.addEventListener("click", function(event) {
            const confirmation = confirm("Are you sure you want to delete this camera?");
            if (!confirmation) {
                event.preventDefault();
            }
        });
    });
});