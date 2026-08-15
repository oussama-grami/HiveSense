import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';
import '../../core/theme/app_theme.dart';
import '../../providers/auth_provider.dart' as ap;

class OtpScreen extends StatefulWidget {
  final String phone;
  final String mode;
  final String? firstName;
  final String? lastName;
  final String? password;
  final String? role;

  const OtpScreen({
    super.key,
    required this.phone,
    required this.mode,
    this.firstName,
    this.lastName,
    this.password,
    this.role,
  });

  @override
  State<OtpScreen> createState() => _OtpScreenState();
}

class _OtpScreenState extends State<OtpScreen> {
  final List<TextEditingController> _controllers =
      List.generate(6, (_) => TextEditingController());
  final List<FocusNode> _focusNodes =
      List.generate(6, (_) => FocusNode());

  int  _secondsLeft = 60;
  bool _canResend   = false;

  @override
  void initState() {
    super.initState();
    _startTimer();
  }

  void _startTimer() {
    _secondsLeft = 60;
    _canResend   = false;
    Future.doWhile(() async {
      await Future.delayed(const Duration(seconds: 1));
      if (!mounted) return false;
      setState(() {
        _secondsLeft--;
        if (_secondsLeft <= 0) _canResend = true;
      });
      return _secondsLeft > 0;
    });
  }

  String get _otp =>
      _controllers.map((c) => c.text).join();

  @override
  void dispose() {
    for (final c in _controllers) c.dispose();
    for (final f in _focusNodes) f.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Consumer<ap.AuthProvider>(
          builder: (context, auth, _) => Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Back button
                GestureDetector(
                  onTap: () => context.go('/login'),
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
                const SizedBox(height: 32),
                // Header
                Container(
                  width: 64, height: 64,
                  decoration: BoxDecoration(
                    color: AppColors.primary.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: const Icon(Icons.sms_rounded,
                      color: AppColors.primary, size: 32),
                ),
                const SizedBox(height: 20),
                const Text('Vérification SMS',
                  style: TextStyle(
                    color: AppColors.textPrimary,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  )),
                const SizedBox(height: 8),
                RichText(
                  text: TextSpan(
                    style: const TextStyle(
                      color: AppColors.textSecondary,
                      fontSize: 14,
                    ),
                    children: [
                      const TextSpan(
                          text: 'Code envoyé au '),
                      TextSpan(
                        text: widget.phone,
                        style: const TextStyle(
                          color: AppColors.primary,
                          fontWeight: FontWeight.bold,
                        )),
                    ],
                  ),
                ),
                const SizedBox(height: 40),
                // Champs OTP
                Row(
                  mainAxisAlignment:
                      MainAxisAlignment.spaceBetween,
                  children: List.generate(6, (i) =>
                    _otpField(i)),
                ),
                const SizedBox(height: 24),
                // Timer
                Center(
                  child: _canResend
                      ? TextButton.icon(
                          onPressed: () async {
                            await auth.sendOTP(widget.phone);
                            _startTimer();
                          },
                          icon: const Icon(
                            Icons.refresh_rounded,
                            color: AppColors.primary,
                            size: 16,
                          ),
                          label: const Text(
                            'Renvoyer le code',
                            style: TextStyle(
                              color: AppColors.primary),
                          ),
                        )
                      : Row(
                          mainAxisAlignment:
                              MainAxisAlignment.center,
                          children: [
                            const Icon(
                              Icons.timer_rounded,
                              color: AppColors.textHint,
                              size: 16,
                            ),
                            const SizedBox(width: 6),
                            Text(
                              'Renvoyer dans $_secondsLeft s',
                              style: const TextStyle(
                                color: AppColors.textHint,
                                fontSize: 13,
                              )),
                          ],
                        ),
                ),
                if (auth.error != null) ...[
                  const SizedBox(height: 12),
                  Container(
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
                          child: Text(auth.error!,
                            style: const TextStyle(
                              color: AppColors.error,
                              fontSize: 12,
                            )),
                        ),
                      ],
                    ),
                  ),
                ],
                const Spacer(),
                // Bouton vérifier
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: auth.isLoading
                        ? null
                        : () => _handleVerify(auth),
                    child: auth.isLoading
                        ? const SizedBox(
                            width: 20, height: 20,
                            child: CircularProgressIndicator(
                              color: Colors.black,
                              strokeWidth: 2,
                            ))
                        : const Text('Vérifier'),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _otpField(int index) {
    return SizedBox(
      width: 48, height: 56,
      child: TextFormField(
        controller:   _controllers[index],
        focusNode:    _focusNodes[index],
        textAlign:    TextAlign.center,
        keyboardType: TextInputType.number,
        maxLength:    1,
        style: const TextStyle(
          color: AppColors.textPrimary,
          fontSize: 22,
          fontWeight: FontWeight.bold,
        ),
        decoration: InputDecoration(
          counterText: '',
          filled: true,
          fillColor: AppColors.surface,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(
                color: AppColors.surfaceLight),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(
                color: AppColors.primary, width: 2),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(
                color: AppColors.surfaceLight),
          ),
        ),
        inputFormatters: [
          FilteringTextInputFormatter.digitsOnly],
        onChanged: (v) {
          if (v.isNotEmpty && index < 5) {
            _focusNodes[index + 1].requestFocus();
          }
          if (v.isEmpty && index > 0) {
            _focusNodes[index - 1].requestFocus();
          }
        },
      ),
    );
  }

  Future<void> _handleVerify(ap.AuthProvider auth) async {
  auth.clearError();

  if (_otp.length < 6) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Entrez le code à 6 chiffres.'),
        backgroundColor: AppColors.error,
      ),
    );
    return;
  }

  final ok = await auth.verifyOTP(_otp);

  if (!mounted) return;

  if (ok) {
    if (widget.mode == 'register') {
      await auth.register(
        phoneNumber: widget.phone,
        password:    widget.password ?? '',
        firstName:   widget.firstName ?? '',
        lastName:    widget.lastName ?? '',
        role:        widget.role ?? 'Apiculteur',
      );
    }
    // Le GoRouterRefreshStream va rediriger automatiquement
    // grâce à authStateChanges — pas besoin de context.go
  }
}
}