module.exports = {
  apps: [
    {
      name: "hyperliquid-bot",
      script: "main.py",
      interpreter: "python3",
      cwd: "/home/ubuntu/hyperliquid-bot",
      watch: false,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      env: {
        PYTHONUNBUFFERED: "1",
      },
      log_file: "/home/ubuntu/hyperliquid-bot/logs/pm2.log",
      out_file: "/home/ubuntu/hyperliquid-bot/logs/pm2-out.log",
      error_file: "/home/ubuntu/hyperliquid-bot/logs/pm2-err.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss",
    },
  ],
};
