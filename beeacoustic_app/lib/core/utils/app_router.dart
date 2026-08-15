import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../../screens/splash_screen.dart';
import '../../screens/auth/login_screen.dart';
import '../../screens/auth/otp_screen.dart';
import '../../screens/profile/profile_screen.dart';
import '../../screens/dashboard/dashboard_screen.dart';
import '../../screens/hive_detail/hive_detail_screen.dart';
import '../../screens/audio_analysis/audio_analysis_screen.dart';
import '../../screens/statistics/statistics_screen.dart';
import '../../screens/alerts/alerts_screen.dart';
import '../../screens/settings/settings_screen.dart';
import '../../widgets/common/main_scaffold.dart';
import '../../providers/auth_provider.dart' as ap;

class GoRouterRefreshStream extends ChangeNotifier {
  GoRouterRefreshStream(Stream<dynamic> stream) {
    notifyListeners();
    stream.listen((_) => notifyListeners());
  }
}

final GoRouter appRouter = GoRouter(
  initialLocation: '/splash',
  refreshListenable: GoRouterRefreshStream(
      FirebaseAuth.instance.authStateChanges()),
  redirect: (context, state) {
    final auth     = context.read<ap.AuthProvider>();
    final loggedIn = auth.isLoggedIn;
    final loc      = state.uri.toString();

    final isPublic = loc.startsWith('/splash') ||
                     loc.startsWith('/login')  ||
                     loc.startsWith('/otp');

    if (!loggedIn && !isPublic) return '/login';
    if (loggedIn  && loc == '/login') return '/';
    return null;
  },
  routes: [
    GoRoute(
      path: '/splash',
      builder: (_, __) => const SplashScreen(),
    ),
    GoRoute(
      path: '/login',
      builder: (_, __) => const LoginScreen(),
    ),
    GoRoute(
      path: '/otp',
      builder: (context, state) {
        final params = state.uri.queryParameters;
        return OtpScreen(
          phone:     Uri.decodeComponent(
              params['phone'] ?? ''),
          mode:      params['mode']      ?? 'login',
          firstName: params['firstName'] != null
              ? Uri.decodeComponent(params['firstName']!)
              : null,
          lastName:  params['lastName'] != null
              ? Uri.decodeComponent(params['lastName']!)
              : null,
          password:  params['password'] != null
              ? Uri.decodeComponent(params['password']!)
              : null,
          role:      params['role'] != null
              ? Uri.decodeComponent(params['role']!)
              : null,
        );
      },
    ),
    GoRoute(
      path: '/profile',
      builder: (_, __) => const ProfileScreen(),
    ),
    ShellRoute(
      builder: (context, state, child) =>
          MainScaffold(child: child),
      routes: [
        GoRoute(
          path: '/',
          builder: (_, __) => const DashboardScreen(),
        ),
        GoRoute(
          path: '/hive/:id',
          builder: (context, state) => HiveDetailScreen(
            hiveId: state.pathParameters['id'] ?? '',
          ),
        ),
        GoRoute(
          path: '/audio',
          builder: (_, __) => const AudioAnalysisScreen(),
        ),
        GoRoute(
          path: '/statistics',
          builder: (_, __) => const StatisticsScreen(),
        ),
        GoRoute(
          path: '/alerts',
          builder: (_, __) => const AlertsScreen(),
        ),
        GoRoute(
          path: '/settings',
          builder: (_, __) => const SettingsScreen(),
        ),
      ],
    ),
  ],
);