import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';
import '../../core/theme/app_theme.dart';
import '../../providers/auth_provider.dart' as ap;

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});
  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  bool _obscurePassword  = true;
  bool _obscurePassword2 = true;

  // Login
  final _phoneLogin    = TextEditingController();
  final _passwordLogin = TextEditingController();

  // Register
  final _phoneRegister    = TextEditingController();
  final _passwordRegister = TextEditingController();
  final _confirmPassword  = TextEditingController();
  final _firstName        = TextEditingController();
  final _lastName         = TextEditingController();
  String _selectedRole    = 'Apiculteur';

  final List<String> _roles = [
    'Apiculteur',
    'Technicien',
    'Gestionnaire',
    'Chercheur',
  ];

  @override
void initState() {
  super.initState();
  _tabController = TabController(length: 2, vsync: this);
  // Réinitialiser l'état auth au chargement
  WidgetsBinding.instance.addPostFrameCallback((_) {
    final auth = context.read<ap.AuthProvider>();
    auth.resetLoading();
  });
}

  @override
  void dispose() {
    _tabController.dispose();
    _phoneLogin.dispose();
    _passwordLogin.dispose();
    _phoneRegister.dispose();
    _passwordRegister.dispose();
    _confirmPassword.dispose();
    _firstName.dispose();
    _lastName.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            children: [
              _buildHeader(),
              _buildTabBar(),
              _buildTabContent(),
            ],
          ),
        ),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(32),
      child: Column(
        children: [
          Container(
            width: 80, height: 80,
            decoration: BoxDecoration(
              color: AppColors.primary,
              borderRadius: BorderRadius.circular(24),
              boxShadow: [
                BoxShadow(
                  color: AppColors.primary.withOpacity(0.4),
                  blurRadius: 20,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: const Icon(Icons.hexagon_rounded,
                color: Colors.black, size: 44),
          ),
          const SizedBox(height: 16),
          const Text('HiveSense',
            style: TextStyle(
              color: AppColors.textPrimary,
              fontSize: 28,
              fontWeight: FontWeight.bold,
            )),
          const SizedBox(height: 4),
          const Text('Surveillance intelligente de vos ruches',
            style: TextStyle(
              color: AppColors.textSecondary,
              fontSize: 13,
            )),
        ],
      ),
    );
  }

  // ── TabBar ────────────────────────────────────────────────────────────────
  Widget _buildTabBar() {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
      ),
      child: TabBar(
        controller: _tabController,
        indicator: BoxDecoration(
          color: AppColors.primary,
          borderRadius: BorderRadius.circular(12),
        ),
        indicatorSize: TabBarIndicatorSize.tab,
        labelColor: Colors.black,
        unselectedLabelColor: AppColors.textSecondary,
        labelStyle: const TextStyle(
            fontWeight: FontWeight.bold, fontSize: 14),
        tabs: const [
          Tab(text: 'Connexion'),
          Tab(text: 'Inscription'),
        ],
      ),
    );
  }

  // ── Tab Content ───────────────────────────────────────────────────────────
  Widget _buildTabContent() {
    return SizedBox(
      height: 600,
      child: TabBarView(
        controller: _tabController,
        children: [
          _buildLoginTab(),
          _buildRegisterTab(),
        ],
      ),
    );
  }

  // ── Onglet Connexion ──────────────────────────────────────────────────────
  Widget _buildLoginTab() {
    return Consumer<ap.AuthProvider>(
      builder: (context, auth, _) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 8),
            const Text('Bon retour !',
              style: TextStyle(
                color: AppColors.textPrimary,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              )),
            const SizedBox(height: 4),
            const Text('Connectez-vous à votre compte',
              style: TextStyle(
                color: AppColors.textSecondary,
                fontSize: 13,
              )),
            const SizedBox(height: 24),
            _phoneField(_phoneLogin, 'Numéro de téléphone'),
            const SizedBox(height: 14),
            _passwordField(_passwordLogin, 'Mot de passe',
                _obscurePassword,
                () => setState(
                    () => _obscurePassword = !_obscurePassword)),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton(
                onPressed: () {},
                child: const Text('Mot de passe oublié ?',
                  style: TextStyle(
                    color: AppColors.primary,
                    fontSize: 12,
                  )),
              ),
            ),
            if (auth.error != null)
              _errorBox(auth.error!),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: auth.isLoading
                    ? null
                    : () => _handleLogin(auth),
                child: auth.isLoading
                    ? const SizedBox(
                        width: 20, height: 20,
                        child: CircularProgressIndicator(
                          color: Colors.black,
                          strokeWidth: 2,
                        ))
                    : const Text('Se connecter'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Onglet Inscription ────────────────────────────────────────────────────
  Widget _buildRegisterTab() {
    return Consumer<ap.AuthProvider>(
      builder: (context, auth, _) => Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Créer un compte',
              style: TextStyle(
                color: AppColors.textPrimary,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              )),
            const SizedBox(height: 4),
            const Text(
              'Un code de vérification sera envoyé par SMS',
              style: TextStyle(
                color: AppColors.textSecondary,
                fontSize: 13,
              )),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(child: _textField(
                    _firstName, 'Prénom', Icons.person_rounded)),
                const SizedBox(width: 12),
                Expanded(child: _textField(
                    _lastName, 'Nom', Icons.person_rounded)),
              ],
            ),
            const SizedBox(height: 12),
            _phoneField(_phoneRegister, 'Numéro de téléphone'),
            const SizedBox(height: 12),
            _passwordField(_passwordRegister, 'Mot de passe',
                _obscurePassword,
                () => setState(
                    () => _obscurePassword = !_obscurePassword)),
            const SizedBox(height: 12),
            _passwordField(_confirmPassword,
                'Confirmer le mot de passe',
                _obscurePassword2,
                () => setState(() =>
                    _obscurePassword2 = !_obscurePassword2)),
            const SizedBox(height: 12),
            _roleDropdown(),
            if (auth.error != null)
              _errorBox(auth.error!),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: auth.isLoading
                    ? null
                    : () => _handleRegister(auth),
                child: auth.isLoading
                    ? const SizedBox(
                        width: 20, height: 20,
                        child: CircularProgressIndicator(
                          color: Colors.black,
                          strokeWidth: 2,
                        ))
                    : const Text('Recevoir le code SMS'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Widgets communs ───────────────────────────────────────────────────────
  Widget _phoneField(TextEditingController ctrl, String hint) {
  return TextFormField(
    controller: ctrl,
    keyboardType: TextInputType.phone,
    style: const TextStyle(color: AppColors.textPrimary),
    decoration: InputDecoration(
      hintText: hint,
      hintStyle: const TextStyle(color: AppColors.textHint),
      prefixIcon: const Icon(Icons.phone_rounded,
          color: AppColors.textHint, size: 20),
      // Plus de prefixText — l'utilisateur tape +21651844856
    ),
  );
}

  Widget _textField(TextEditingController ctrl,
      String hint, IconData icon) {
    return TextFormField(
      controller: ctrl,
      style: const TextStyle(color: AppColors.textPrimary),
      decoration: InputDecoration(
        hintText: hint,
        prefixIcon: Icon(icon,
            color: AppColors.textHint, size: 20),
      ),
    );
  }

  Widget _passwordField(TextEditingController ctrl,
      String hint, bool obscure, VoidCallback toggle) {
    return TextFormField(
      controller: ctrl,
      obscureText: obscure,
      style: const TextStyle(color: AppColors.textPrimary),
      decoration: InputDecoration(
        hintText: hint,
        prefixIcon: const Icon(Icons.lock_rounded,
            color: AppColors.textHint, size: 20),
        suffixIcon: IconButton(
          icon: Icon(
            obscure
                ? Icons.visibility_rounded
                : Icons.visibility_off_rounded,
            color: AppColors.textHint, size: 20,
          ),
          onPressed: toggle,
        ),
      ),
    );
  }

  Widget _roleDropdown() {
    return DropdownButtonFormField<String>(
      value: _selectedRole,
      dropdownColor: AppColors.surfaceLight,
      style: const TextStyle(color: AppColors.textPrimary),
      decoration: const InputDecoration(
        hintText: 'Rôle',
        prefixIcon: Icon(Icons.work_rounded,
            color: AppColors.textHint, size: 20),
      ),
      items: _roles.map((r) => DropdownMenuItem(
        value: r,
        child: Text(r),
      )).toList(),
      onChanged: (v) =>
          setState(() => _selectedRole = v ?? _selectedRole),
    );
  }

  Widget _errorBox(String message) {
    return Container(
      margin: const EdgeInsets.only(top: 10),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: AppColors.error.withOpacity(0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
            color: AppColors.error.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_rounded,
              color: AppColors.error, size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Text(message,
              style: const TextStyle(
                color: AppColors.error, fontSize: 12)),
          ),
        ],
      ),
    );
  }

  // ── Actions ───────────────────────────────────────────────────────────────
  Future<void> _handleLogin(ap.AuthProvider auth) async {
  auth.clearError();

  if (_phoneLogin.text.trim().isEmpty) {
    _showError('Le numéro de téléphone est obligatoire.');
    return;
  }
  if (_passwordLogin.text.isEmpty) {
    _showError('Le mot de passe est obligatoire.');
    return;
  }

  final raw   = _phoneLogin.text.trim()
      .replaceAll(' ', '')
      .replaceAll('-', '');
  final phone = raw.startsWith('+') ? raw : '+216$raw';

  print('Numéro login : "$phone"');

  final ok = await auth.login(phone, _passwordLogin.text.trim());
  if (ok && mounted) {
    await auth.sendOTP(phone);
    if (mounted) context.go('/otp?phone=${Uri.encodeComponent(phone)}&mode=login');
  }
}

Future<void> _handleRegister(ap.AuthProvider auth) async {
  auth.clearError();

  // Validation des champs
  if (_firstName.text.trim().isEmpty) {
    _showError('Le prénom est obligatoire.');
    return;
  }
  if (_lastName.text.trim().isEmpty) {
    _showError('Le nom est obligatoire.');
    return;
  }
  if (_phoneRegister.text.trim().isEmpty) {
    _showError('Le numéro de téléphone est obligatoire.');
    return;
  }
  if (_phoneRegister.text.trim().replaceAll(' ', '').length < 8) {
    _showError('Le numéro doit contenir au moins 8 chiffres.');
    return;
  }
  if (_passwordRegister.text.isEmpty) {
    _showError('Le mot de passe est obligatoire.');
    return;
  }
  if (_passwordRegister.text.length < 6) {
    _showError('Le mot de passe doit contenir au moins 6 caractères.');
    return;
  }
  if (_passwordRegister.text != _confirmPassword.text) {
    _showError('Les mots de passe ne correspondent pas.');
    return;
  }

  final raw   = _phoneRegister.text.trim()
      .replaceAll(' ', '')
      .replaceAll('-', '');
  final phone = raw.startsWith('+') ? raw : '+216$raw';

  print('Numéro inscription : "$phone"');

  final ok = await auth.sendOTP(phone, checkExisting: true);
  if (ok && mounted) {
    context.go(
      '/otp'
      '?phone=${Uri.encodeComponent(phone)}'
      '&mode=register'
      '&firstName=${Uri.encodeComponent(_firstName.text.trim())}'
      '&lastName=${Uri.encodeComponent(_lastName.text.trim())}'
      '&password=${Uri.encodeComponent(_passwordRegister.text)}'
      '&role=${Uri.encodeComponent(_selectedRole)}',
    );
  }
}

void _showError(String message) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      content: Row(
        children: [
          const Icon(Icons.error_rounded,
              color: Colors.white, size: 18),
          const SizedBox(width: 8),
          Expanded(child: Text(message)),
        ],
      ),
      backgroundColor: AppColors.error,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12)),
    ),
  );
}
}