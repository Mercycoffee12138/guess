#!/bin/sh
skip=49

tab='	'
nl='
'
IFS=" $tab$nl"

umask=`umask`
umask 77

gztmpdir=
trap 'res=$?
  test -n "$gztmpdir" && rm -fr "$gztmpdir"
  (exit $res); exit $res
' 0 1 2 3 5 10 13 15

case $TMPDIR in
  / | /*/) ;;
  /*) TMPDIR=$TMPDIR/;;
  *) TMPDIR=/tmp/;;
esac
if type mktemp >/dev/null 2>&1; then
  gztmpdir=`mktemp -d "${TMPDIR}gztmpXXXXXXXXX"`
else
  gztmpdir=${TMPDIR}gztmp$$; mkdir $gztmpdir
fi || { (exit 127); exit 127; }

gztmp=$gztmpdir/$0
case $0 in
-* | */*'
') mkdir -p "$gztmp" && rm -r "$gztmp";;
*/*) gztmp=$gztmpdir/`basename "$0"`;;
esac || { (exit 127); exit 127; }

case `printf 'X\n' | tail -n +1 2>/dev/null` in
X) tail_n=-n;;
*) tail_n=;;
esac
if tail $tail_n +$skip <"$0" | gzip -cd > "$gztmp"; then
  umask $umask
  chmod 700 "$gztmp"
  (sleep 5; rm -fr "$gztmpdir") 2>/dev/null &
  "$gztmp" ${1+"$@"}; res=$?
else
  printf >&2 '%s\n' "Cannot decompress $0"
  (exit 127); res=127
fi; exit $res
�#�htest.sh �T]O�V�?���T��$�LA�Z�X�J���|B<9v;0)�4A�tCM���aK�Ti�ieğ��䊿���M(��ܜ��y?��y�	݊$U=���4	��)��a`����5������fa������O>53 �@�,������:\t.m�`����C�qb���Ν?w��N�a7J���/��Z#)���Ѹ8=5�� :.��XQ ����߬��-����D2@B���w��m�%w�?l��"s��q���v=أsPj��B�pw��S=�ϗY���/_�� .D�rN�4�%ҋ��<5͈X�.��`���0md(~-�8���^E���<��s@�G��=tn���z�����3��}�Xsv��k�v�x~EE?=�/pc0O�\:ŢC�z}OީT��5�����*�b���U���l=k��\��(t)R�`=�)��)��'���N�s�����&N='�IU��}ߚ�$��P����2��Y=.8=���p����j*�̨�E�4��Zi�'�M��V�ʑ�b�^;b����j{�U|(u�7X���F�b'�jr֤J<JTݢ�Y��J�AR�fRBӪFQ�[7�2�o���g�v�~)f�bp�M��,�A5��	��/�����(�N;���S4����#+��ݣy^n��c�v�L�sv�XwOrvwp\�W�ݝ���`o�C���Y������o���.myE�Yߨ q��_��QG�-̓�[	����k��?���ԇS6W����ظ�*�Î_���W�
c����6�v��`�j�J�$*
��<��{�ڪ�8���?<C��g���}��T^���7�����y*��EqF�͙��rT��%I�-
w���R8#�)�@
)��P#͘O0�^�o�?�	�@�@pu�ڳ�Ĥ��j ��^���HD�=�i�1�h>4|A^�rRcV�B�,��|/��\R�Î�?J��cR��L("����  