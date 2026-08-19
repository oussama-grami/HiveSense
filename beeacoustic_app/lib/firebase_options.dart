import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) return web;
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      default:
        return web;
    }
  }

  // ⚠️ Remplace ces valeurs par celles de ton firebaseConfig
  static const FirebaseOptions web = FirebaseOptions(
    apiKey:            "AIzaSyBaSZiUgILfjtWXijzTlq3bDHAfccXEzNQ",
    authDomain:        "hivesense-a6961.firebaseapp.com",
    projectId:         "hivesense-a6961",
    storageBucket:     "hivesense-a6961.firebasestorage.app",
    messagingSenderId: "251433416356",
    appId:             "1:251433416356:web:12b3a285e81beaa0a7d831",
  );

  static const FirebaseOptions android = FirebaseOptions(
    apiKey:            "AIzaSyBaSZiUgILfjtWXijzTlq3bDHAfccXEzNQ",
    authDomain:        "hivesense-a6961.firebaseapp.com",
    projectId:         "hivesense-a6961",
    storageBucket:     "hivesense-a6961.firebasestorage.app",
    messagingSenderId: "251433416356",
    appId:             "1:251433416356:web:12b3a285e81beaa0a7d831",
  );
}