// ─── ClinicalML Authentication Module ────────────────────────────────────────
// Uses Firebase Authentication (Email/Password + Google OAuth)
// This file is loaded as a module in login.html and index.html
// ─────────────────────────────────────────────────────────────────────────────

import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.0/firebase-app.js";
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  signOut as firebaseSignOut,
  onAuthStateChanged,
  updateProfile
} from "https://www.gstatic.com/firebasejs/11.0.0/firebase-auth.js";

// ── Initialize Firebase ──────────────────────────────────────────────────────
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

// ── Friendly Error Messages ──────────────────────────────────────────────────
const AUTH_ERRORS = {
  'auth/email-already-in-use': 'This email is already registered. Try logging in instead.',
  'auth/invalid-email': 'Please enter a valid email address.',
  'auth/operation-not-allowed': 'This sign-in method is not enabled. Contact admin.',
  'auth/weak-password': 'Password must be at least 6 characters.',
  'auth/user-disabled': 'This account has been disabled. Contact admin.',
  'auth/user-not-found': 'No account found with this email. Please register first.',
  'auth/wrong-password': 'Incorrect password. Please try again.',
  'auth/invalid-credential': 'Invalid email or password. Please try again.',
  'auth/too-many-requests': 'Too many failed attempts. Please wait and try again.',
  'auth/popup-closed-by-user': 'Google sign-in was cancelled.',
  'auth/network-request-failed': 'Network error. Check your internet connection.',
  'auth/popup-blocked': 'Popup was blocked. Allow popups for this site.',
};

function getErrorMessage(error) {
  return AUTH_ERRORS[error.code] || error.message || 'An unexpected error occurred.';
}

// ── Auth Functions ───────────────────────────────────────────────────────────

/** Register a new user with email, password, and display name */
async function registerUser(email, password, displayName) {
  const cred = await createUserWithEmailAndPassword(auth, email, password);
  await updateProfile(cred.user, { displayName });
  return cred.user;
}

/** Sign in with email and password */
async function loginUser(email, password) {
  const cred = await signInWithEmailAndPassword(auth, email, password);
  return cred.user;
}

/** Sign in with Google OAuth popup */
async function loginWithGoogle() {
  const cred = await signInWithPopup(auth, googleProvider);
  return cred.user;
}

/** Sign out and redirect to login page */
async function logoutUser() {
  await firebaseSignOut(auth);
  window.location.href = 'login.html';
}

/** Get the currently signed-in user (or null) */
function getCurrentUser() {
  return auth.currentUser;
}

// ── Auth State Observer ──────────────────────────────────────────────────────

/**
 * Watch auth state. Calls callback(user) when state changes.
 * Returns the unsubscribe function.
 */
function watchAuthState(callback) {
  return onAuthStateChanged(auth, callback);
}

/**
 * Guard function for protected pages.
 * Redirects to login.html if not authenticated.
 * Calls onReady(user) when authenticated.
 */
function requireAuth(onReady) {
  onAuthStateChanged(auth, (user) => {
    if (!user) {
      window.location.href = 'login.html';
    } else {
      if (onReady) onReady(user);
    }
  });
}

/**
 * Guard for login page.
 * Redirects to index.html if already authenticated.
 * Calls onReady() when confirmed not authenticated.
 */
function redirectIfLoggedIn(onReady) {
  onAuthStateChanged(auth, (user) => {
    if (user) {
      window.location.href = 'index.html';
    } else {
      if (onReady) onReady();
    }
  });
}

// ── Export for use ───────────────────────────────────────────────────────────
window.ClinicalAuth = {
  registerUser,
  loginUser,
  loginWithGoogle,
  logoutUser,
  getCurrentUser,
  watchAuthState,
  requireAuth,
  redirectIfLoggedIn,
  getErrorMessage
};
