#!/bin/bash
# check permission
# [ "$(whoami)" != "root" ] && exec sudo -- "$0" "$@"
if [ "$(whoami)" == "root" ]; then
  echo "Exiting... DO NOT run as root"
  exit 1
else
  echo "Setup as $(whoami)"
fi

while true; do
  read -r -e -t 10 -p "clean old installation? [y/n] default [y]: " -i "y" yn
  case $yn in
      [Yy]* )
        echo "clean old installation... "
        screen -S Krypton -X quit
        sudo rm -r Krypton
        sudo apt purge redis redis-tools -y -qq > /dev/null 2> /dev/null
        sudo rm -r /etc/redis
        echo "Done"
        break
        ;;
      [Nn]* )
        break
        ;;
      * )
        echo "Please answer yes or no."
        ;;
  esac
done

# install packages
echo -n "setup requirements...  "
sudo apt update -qq > /dev/null 2> /dev/null
sudo apt install python3.8 python3.8-venv python3.8-dev git redis -y -qq > /dev/null 2> /dev/null
# clone the project
git clone https://github.com/BolunHan/Krypton.git --quiet
# make venv
cd Krypton || exit 1
python3.8 -m venv venv
. venv/bin/activate
python -m pip -q install -U setuptools pip wheel
pip -q install -r requirements.txt
echo "Done"
# config redis
echo -n "config redis...  "
sudo systemctl stop redis-server
sudo sed -i 's/supervised no/supervised systemd/' /etc/redis/redis.conf
sudo sed -i 's/bind 127.0.0.1 ::1/# bind 127.0.0.1 ::1/' /etc/redis/redis.conf
sudo sed -i 's/port 6379/port 13168/' /etc/redis/redis.conf
sudo sed -i 's/save 900 1/# save 900 1/' /etc/redis/redis.conf
sudo sed -i 's/save 300 10/# save 300 10/' /etc/redis/redis.conf
sudo sed -i 's/save 60 10000/# save 60 10000/' /etc/redis/redis.conf
sudo sed -i 's/aof-use-rdb-preamble yes/aof-use-rdb-preamble no/' /etc/redis/redis.conf
sudo sed -i 's/# requirepass foobared/requirepass kBSVQUul+EglrLu21PKShlcoBfkK6kkymM6ZDq6nGqjhI5xeXF6W1a6aJIGu3SW\/MILbO7r+iottdD+H/' /etc/redis/redis.conf
sudo systemctl restart redis.service
echo "Done"
echo "Setup complete! Firewall port 13168 must be manually opened to subscribe data."
screen -S Krypton.Huobi.Spot -d -m env KRYPTON_CWD=Huobi python Krypton/Relay/Huobi.Spot.py
screen -S Krypton.Binance.Spot -d -m env KRYPTON_CWD=Binance python Krypton/Relay/Binance.Spot.py
echo "Relay service start!"
exit 0
