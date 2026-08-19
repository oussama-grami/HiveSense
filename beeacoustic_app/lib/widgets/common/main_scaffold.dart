import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../../core/theme/app_theme.dart';

class MainScaffold extends StatelessWidget {
  final Widget child;
  const MainScaffold({super.key, required this.child});

  int _selectedIndex(BuildContext context) {
  final location = GoRouterState.of(context).uri.toString();
  if (location == '/')               return 0;
  if (location == '/audio')          return 1;
  if (location == '/statistics')     return 2;
  if (location == '/alerts')         return 3;
  if (location == '/settings')       return 4;
  if (location.startsWith('/hive'))  return 0;
  return 0;
}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: child,
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: AppColors.surface,
          border: Border(
            top: BorderSide(color: AppColors.surfaceLight, width: 1),
          ),
        ),
        child: BottomNavigationBar(
          currentIndex: _selectedIndex(context),
        onTap: (index) {
            switch (index) {
              case 0: context.go('/');           break;
              case 1: context.go('/audio');      break;
              case 2: context.go('/statistics'); break;
              case 3: context.go('/alerts');     break;
              case 4: context.go('/settings');   break;
            }
          },
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.dashboard_rounded),
              label: 'Tableau de bord',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.mic_rounded),
              label: 'Analyse',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.bar_chart_rounded),
              label: 'Statistiques',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.notifications_rounded),
              label: 'Alertes',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.settings_rounded),
              label: 'Paramètres',
            ),
          ],
        ),
      ),
    );
  }
}