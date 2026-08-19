import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';
import '../../core/theme/app_theme.dart';
import '../../providers/auth_provider.dart' as ap;

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});
  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  Map<String, dynamic>? _profile;
  bool _isEditing = false;
  bool _isLoading = false;

  final _firstNameCtrl = TextEditingController();
  final _lastNameCtrl  = TextEditingController();
  final _roleCtrl      = TextEditingController();

  final List<String> _roles = [
    'Apiculteur', 'Technicien', 'Gestionnaire', 'Chercheur'];
  String _selectedRole = 'Apiculteur';

  @override
  void initState() {
    super.initState();
    _loadProfile();
  }

  Future<void> _loadProfile() async {
    setState(() => _isLoading = true);
    final auth    = context.read<ap.AuthProvider>();
    final profile = await auth.getUserProfile();
    setState(() {
      _profile  = profile;
      _isLoading = false;
      if (profile != null) {
        _firstNameCtrl.text =
            profile['firstName'] ?? '';
        _lastNameCtrl.text  =
            profile['lastName'] ?? '';
        _selectedRole =
            profile['role'] ?? 'Apiculteur';
      }
    });
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final image  = await picker.pickImage(
        source: ImageSource.gallery);
    if (image != null) {
      // Pour l'instant on affiche juste un message
      // En production : uploader sur Firebase Storage
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Photo sélectionnée — upload disponible en production'),
          backgroundColor: AppColors.info,
        ),
      );
    }
  }

  @override
  void dispose() {
    _firstNameCtrl.dispose();
    _lastNameCtrl.dispose();
    _roleCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: _isLoading
            ? const Center(child: CircularProgressIndicator(
                color: AppColors.primary))
            : SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    _buildHeader(context),
                    const SizedBox(height: 24),
                    _buildAvatar(),
                    const SizedBox(height: 24),
                    _buildInfoSection(),
                    const SizedBox(height: 16),
                    _buildStatsSection(),
                    const SizedBox(height: 16),
                    _buildActionsSection(context),
                    const SizedBox(height: 80),
                  ],
                ),
              ),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────
  Widget _buildHeader(BuildContext context) {
    return Row(
      children: [
        GestureDetector(
          onTap: () => context.go('/'),
          child: Container(
            width: 36, height: 36,
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(Icons.arrow_back_rounded,
                color: AppColors.textPrimary, size: 20),
          ),
        ),
        const SizedBox(width: 12),
        const Expanded(
          child: Text('Mon Profil',
            style: TextStyle(
              color: AppColors.textPrimary,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            )),
        ),
        GestureDetector(
          onTap: () {
            if (_isEditing) _saveProfile();
            setState(() => _isEditing = !_isEditing);
          },
          child: Container(
            padding: const EdgeInsets.symmetric(
                horizontal: 14, vertical: 8),
            decoration: BoxDecoration(
              color: _isEditing
                  ? AppColors.primary
                  : AppColors.surface,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(
              _isEditing ? 'Sauvegarder' : 'Modifier',
              style: TextStyle(
                color: _isEditing
                    ? Colors.black
                    : AppColors.primary,
                fontSize: 13,
                fontWeight: FontWeight.bold,
              )),
          ),
        ),
      ],
    );
  }

  // ── Avatar ────────────────────────────────────────────────────────────────
  Widget _buildAvatar() {
    final name = '${_profile?['firstName'] ?? ''} '
        '${_profile?['lastName'] ?? ''}';
    final initials = name.trim().isEmpty
        ? '?'
        : name.trim().split(' ').map((w) =>
            w.isNotEmpty ? w[0] : '').take(2).join();

    return Column(
      children: [
        Stack(
          children: [
            Container(
              width: 100, height: 100,
              decoration: BoxDecoration(
                color: AppColors.primary.withOpacity(0.2),
                shape: BoxShape.circle,
                border: Border.all(
                    color: AppColors.primary, width: 3),
              ),
              child: Center(
                child: Text(
                  initials.toUpperCase(),
                  style: const TextStyle(
                    color: AppColors.primary,
                    fontSize: 36,
                    fontWeight: FontWeight.bold,
                  )),
              ),
            ),
            if (_isEditing)
              Positioned(
                right: 0, bottom: 0,
                child: GestureDetector(
                  onTap: _pickImage,
                  child: Container(
                    width: 32, height: 32,
                    decoration: const BoxDecoration(
                      color: AppColors.primary,
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.camera_alt_rounded,
                      color: Colors.black,
                      size: 16,
                    ),
                  ),
                ),
              ),
          ],
        ),
        const SizedBox(height: 12),
        Text(
          name.trim().isEmpty ? 'Utilisateur' : name.trim(),
          style: const TextStyle(
            color: AppColors.textPrimary,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          )),
        const SizedBox(height: 4),
        Container(
          padding: const EdgeInsets.symmetric(
              horizontal: 12, vertical: 4),
          decoration: BoxDecoration(
            color: AppColors.primary.withOpacity(0.15),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            _profile?['role'] ?? 'Apiculteur',
            style: const TextStyle(
              color: AppColors.primary,
              fontSize: 12,
              fontWeight: FontWeight.bold,
            )),
        ),
      ],
    );
  }

  // ── Informations ──────────────────────────────────────────────────────────
  Widget _buildInfoSection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.person_rounded,
                  color: AppColors.primary, size: 18),
              SizedBox(width: 8),
              Text('Informations personnelles',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                )),
            ],
          ),
          const SizedBox(height: 16),
          _isEditing
              ? Column(
                  children: [
                    Row(
                      children: [
                        Expanded(child: _editField(
                            _firstNameCtrl, 'Prénom')),
                        const SizedBox(width: 12),
                        Expanded(child: _editField(
                            _lastNameCtrl, 'Nom')),
                      ],
                    ),
                    const SizedBox(height: 12),
                    DropdownButtonFormField<String>(
                      value: _selectedRole,
                      dropdownColor: AppColors.surfaceLight,
                      style: const TextStyle(
                          color: AppColors.textPrimary),
                      decoration: const InputDecoration(
                        hintText: 'Rôle',
                        prefixIcon: Icon(Icons.work_rounded,
                            color: AppColors.textHint,
                            size: 20),
                      ),
                      items: _roles.map((r) =>
                          DropdownMenuItem(
                              value: r,
                              child: Text(r))).toList(),
                      onChanged: (v) => setState(
                          () => _selectedRole =
                              v ?? _selectedRole),
                    ),
                  ],
                )
              : Column(
                  children: [
                    _infoRow(Icons.badge_rounded,
                        'Prénom',
                        _profile?['firstName'] ?? '—'),
                    _divider(),
                    _infoRow(Icons.badge_rounded,
                        'Nom',
                        _profile?['lastName'] ?? '—'),
                    _divider(),
                    _infoRow(Icons.phone_rounded,
                        'Téléphone',
                        _profile?['phoneNumber'] ?? '—'),
                    _divider(),
                    _infoRow(Icons.work_rounded,
                        'Rôle',
                        _profile?['role'] ?? '—'),
                  ],
                ),
        ],
      ),
    );
  }

  // ── Statistiques ──────────────────────────────────────────────────────────
  Widget _buildStatsSection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.bar_chart_rounded,
                  color: AppColors.primary, size: 18),
              SizedBox(width: 8),
              Text('Statistiques',
                style: TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                )),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(child: _statCard(
                  'Ruches gérées',
                  '${_profile?['hivesCount'] ?? 3}',
                  Icons.hive_rounded,
                  AppColors.primary)),
              const SizedBox(width: 12),
              Expanded(child: _statCard(
                  'Alertes reçues',
                  '12',
                  Icons.notifications_rounded,
                  AppColors.warning)),
              const SizedBox(width: 12),
              Expanded(child: _statCard(
                  'Analyses',
                  '47',
                  Icons.psychology_rounded,
                  AppColors.success)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _statCard(String label, String value,
      IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surfaceLight,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Icon(icon, color: color, size: 22),
          const SizedBox(height: 6),
          Text(value,
            style: TextStyle(
              color: color,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            )),
          Text(label,
            style: const TextStyle(
              color: AppColors.textHint,
              fontSize: 10,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  // ── Actions ───────────────────────────────────────────────────────────────
  Widget _buildActionsSection(BuildContext context) {
    return Column(
      children: [
        _actionTile(
          'Changer le mot de passe',
          Icons.lock_rounded,
          AppColors.info,
          () {},
        ),
        const SizedBox(height: 8),
        _actionTile(
          'Notifications',
          Icons.notifications_rounded,
          AppColors.warning,
          () => context.go('/settings'),
        ),
        const SizedBox(height: 8),
        _actionTile(
          'À propos de HiveSense',
          Icons.info_rounded,
          AppColors.textSecondary,
          () {},
        ),
        const SizedBox(height: 16),
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            onPressed: () => _handleSignOut(context),
            icon: const Icon(Icons.logout_rounded,
                color: AppColors.error),
            label: const Text('Se déconnecter',
              style: TextStyle(color: AppColors.error)),
            style: OutlinedButton.styleFrom(
              side: const BorderSide(
                  color: AppColors.error),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
              padding: const EdgeInsets.symmetric(
                  vertical: 14),
            ),
          ),
        ),
      ],
    );
  }

  Widget _actionTile(String label, IconData icon,
      Color color, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(width: 12),
            Expanded(
              child: Text(label,
                style: const TextStyle(
                  color: AppColors.textPrimary,
                  fontSize: 14,
                ))),
            const Icon(Icons.chevron_right_rounded,
                color: AppColors.textHint, size: 18),
          ],
        ),
      ),
    );
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  Widget _infoRow(IconData icon, String label,
      String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(icon, color: AppColors.textHint, size: 18),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label,
                  style: const TextStyle(
                    color: AppColors.textHint,
                    fontSize: 11,
                  )),
                Text(value,
                  style: const TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 14,
                  )),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _divider() => const Divider(
      height: 1, color: AppColors.surfaceLight);

  Widget _editField(TextEditingController ctrl,
      String hint) {
    return TextFormField(
      controller: ctrl,
      style: const TextStyle(color: AppColors.textPrimary),
      decoration: InputDecoration(hintText: hint),
    );
  }

  Future<void> _saveProfile() async {
    final auth = context.read<ap.AuthProvider>();
    await auth.updateProfile({
      'firstName': _firstNameCtrl.text,
      'lastName':  _lastNameCtrl.text,
      'role':      _selectedRole,
    });
    await _loadProfile();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Profil mis à jour'),
          backgroundColor: AppColors.success,
        ),
      );
    }
  }

  Future<void> _handleSignOut(BuildContext context) async {
    final auth = context.read<ap.AuthProvider>();
    await auth.signOut();
    if (mounted) context.go('/login');
  }
}