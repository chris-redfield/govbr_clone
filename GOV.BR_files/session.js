import { cookiesByName } from './encrypted-local-storage/cookies.js';
import { EncryptedLocalStorage } from './encrypted-local-storage/encrypted-local-storage.js'

const xsrfTokenCookie = cookiesByName()?.["XSRF-TOKEN"];

const storage = new EncryptedLocalStorage(null, { useCookies: true });
let session;
try {
    session = await storage.getItem('GovBr-Portal-Session');
} catch(e) {}

const barraGovbr = document.querySelector("#barra-govbr");
!!session && !!xsrfTokenCookie && document.dispatchEvent(new CustomEvent("session-found", { bubbles: false, cancelable: true, composed: false, detail: {...session}} ))
barraGovbr.logado = !!session && !!xsrfTokenCookie;

