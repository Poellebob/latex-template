lang="$1"
prv="$PWD"

mkdir -p "$HOME/.local/share/nvim/site/spell"
cd "$HOME/.local/share/nvim/site/spell" || exit 1

curl -O "https://mirrors.kernel.org/vim/runtime/spell/${lang}.utf-8.spl"
curl -O "https://mirrors.kernel.org/vim/runtime/spell/${lang}.utf-8.sug"

cd "$prv" || exit 1
