import 'package:flutter/material.dart';
import '../../core/theme/app_theme.dart';

class SensorCard extends StatelessWidget {
  final String   label;
  final String   value;
  final IconData icon;
  final Color    color;
  final String?  subtitle;
  final double?  progress;

  const SensorCard({
    super.key,
    required this.label,
    required this.value,
    required this.icon,
    required this.color,
    this.subtitle,
    this.progress,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
            color: color.withOpacity(0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 18),
              const Spacer(),
              if (progress != null)
                SizedBox(
                  width: 32, height: 32,
                  child: CircularProgressIndicator(
                    value: progress,
                    backgroundColor:
                        AppColors.surfaceLight,
                    valueColor:
                        AlwaysStoppedAnimation<Color>(color),
                    strokeWidth: 3,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 10),
          Text(value,
            style: TextStyle(
              color: color,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            )),
          Text(label,
            style: const TextStyle(
              color: AppColors.textSecondary,
              fontSize: 11,
            )),
          if (subtitle != null) ...[
            const SizedBox(height: 2),
            Text(subtitle!,
              style: const TextStyle(
                color: AppColors.textHint,
                fontSize: 10,
              )),
          ],
        ],
      ),
    );
  }
}