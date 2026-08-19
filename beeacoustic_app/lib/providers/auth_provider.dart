import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class AuthProvider extends ChangeNotifier {
  final FirebaseAuth      _auth      = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  User?               _user;
  bool                _isLoading         = false;
  String?             _error;
  ConfirmationResult? _confirmationResult;

  // ── Getters ───────────────────────────────────────────────────────────────
  User?   get user       => _user;
  bool    get isLoading  => _isLoading;
  String? get error      => _error;
  bool    get isLoggedIn => _user != null;

  // ── Init ──────────────────────────────────────────────────────────────────
  AuthProvider() {
    _auth.authStateChanges().listen((user) {
      _user = user;
      notifyListeners();
    });
  }

  // ── Envoyer OTP (Flutter Web) ─────────────────────────────────────────────
  Future<bool> sendOTP(String phoneNumber) async {
  _setLoading(true);
  _error = null;

  try {
    _confirmationResult = await FirebaseAuth.instance
        .signInWithPhoneNumber(phoneNumber);

    _setLoading(false);
    return true;
  } on FirebaseAuthException catch (e) {
    print('Code Firebase : ${e.code}');
    print('Message Firebase : ${e.message}');
    _error = '${e.code} : ${e.message}';
    _setLoading(false);
    return false;
  } catch (e) {
    print('Erreur inattendue : $e');
    _error = e.toString();
    _setLoading(false);
    return false;
  }
}

  // ── Vérifier OTP ──────────────────────────────────────────────────────────
  Future<bool> verifyOTP(String otp) async {
    if (_confirmationResult == null) {
      _error = 'Session expirée. Renvoyez le code.';
      notifyListeners();
      return false;
    }

    _setLoading(true);
    _error = null;

    try {
      final result = await _confirmationResult!.confirm(otp);
      _user = result.user;
      _setLoading(false);
      return true;
    } on FirebaseAuthException catch (e) {
      print('Code Firebase : ${e.code}');
      print('Message Firebase : ${e.message}');
      _error = _parseError(e.code);
      _setLoading(false);
      return false;
    } catch (e) {
      print('Erreur inattendue : $e');
      _error = e.toString();
      _setLoading(false);
      return false;
    }
  }

  // ── Connexion ─────────────────────────────────────────────────────────────
  Future<bool> login(String phoneNumber, String password) async {
    _setLoading(true);
    _error = null;

    try {
      final query = await _firestore
          .collection('users')
          .where('phoneNumber', isEqualTo: phoneNumber)
          .where('password',    isEqualTo: password)
          .get();

      if (query.docs.isEmpty) {
        _error = 'Numéro ou mot de passe incorrect.';
        _setLoading(false);
        return false;
      }

      _setLoading(false);
      return true;
    } on FirebaseAuthException catch (e) {
      _error = _parseError(e.code);
      _setLoading(false);
      return false;
    } catch (e) {
      print('Erreur login : $e');
      _error = 'Erreur de connexion.';
      _setLoading(false);
      return false;
    }
  }

  // ── Inscription ───────────────────────────────────────────────────────────
  Future<bool> register({
    required String phoneNumber,
    required String password,
    required String firstName,
    required String lastName,
    required String role,
  }) async {
    _setLoading(true);
    _error = null;

    try {
      if (_user != null) {
        await _firestore
            .collection('users')
            .doc(_user!.uid)
            .set({
          'phoneNumber': phoneNumber,
          'firstName':   firstName,
          'lastName':    lastName,
          'role':        role,
          'password':    password,
          'createdAt':   FieldValue.serverTimestamp(),
          'photoUrl':    null,
          'hivesCount':  0,
        });
      }
      _setLoading(false);
      return true;
    } catch (e) {
      print('Erreur register : $e');
      _error = 'Erreur lors de la création du compte.';
      _setLoading(false);
      return false;
    }
  }

  // ── Profil ────────────────────────────────────────────────────────────────
  Future<Map<String, dynamic>?> getUserProfile() async {
    if (_user == null) return null;
    try {
      final doc = await _firestore
          .collection('users')
          .doc(_user!.uid)
          .get();
      return doc.data();
    } catch (e) {
      print('Erreur getUserProfile : $e');
      return null;
    }
  }

  Future<bool> updateProfile(Map<String, dynamic> data) async {
    if (_user == null) return false;
    _setLoading(true);
    try {
      await _firestore
          .collection('users')
          .doc(_user!.uid)
          .update(data);
      _setLoading(false);
      notifyListeners();
      return true;
    } catch (e) {
      print('Erreur updateProfile : $e');
      _setLoading(false);
      return false;
    }
  }

  // ── Déconnexion ───────────────────────────────────────────────────────────
  Future<void> signOut() async {
    await _auth.signOut();
    _user = null;
    notifyListeners();
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  void _setLoading(bool v) {
    _isLoading = v;
    notifyListeners();
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }

  String _parseError(String code) {
    switch (code) {
      case 'invalid-phone-number':
        return 'Numéro de téléphone invalide.';
      case 'too-many-requests':
        return 'Trop de tentatives. Réessayez plus tard.';
      case 'invalid-verification-code':
        return 'Code incorrect. Vérifiez et réessayez.';
      case 'session-expired':
        return 'Session expirée. Renvoyez le code.';
      case 'quota-exceeded':
        return 'Quota SMS dépassé. Utilisez un numéro de test.';
      default:
        return 'Erreur : $code';
    }
  }
}