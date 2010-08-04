// Inferno utils/8c/8.out.h
// http://code.google.com/p/inferno-os/source/browse/utils/8c/8.out.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#define	NSYM	50
#define	NSNAME	8
#define NOPROF	(1<<0)
#define DUPOK	(1<<1)
#define NOSPLIT	(1<<2)

enum	as
{
	AXXX,
	AAAA,
	AAAD,
	AAAM,
	AAAS,
	AADCB,
	AADCL,
	AADCW,
	AADDB,
	AADDL,
	AADDW,
	AADJSP,
	AANDB,
	AANDL,
	AANDW,
	AARPL,
	ABOUNDL,
	ABOUNDW,
	ABSFL,
	ABSFW,
	ABSRL,
	ABSRW,
	ABTL,
	ABTW,
	ABTCL,
	ABTCW,
	ABTRL,
	ABTRW,
	ABTSL,
	ABTSW,
	ABYTE,
	ACALL,
	ACLC,
	ACLD,
	ACLI,
	ACLTS,
	ACMC,
	ACMPB,
	ACMPL,
	ACMPW,
	ACMPSB,
	ACMPSL,
	ACMPSW,
	ADAA,
	ADAS,
	ADATA,
	ADECB,
	ADECL,
	ADECW,
	ADIVB,
	ADIVL,
	ADIVW,
	AENTER,
	AGLOBL,
	AGOK,
	AHISTORY,
	AHLT,
	AIDIVB,
	AIDIVL,
	AIDIVW,
	AIMULB,
	AIMULL,
	AIMULW,
	AINB,
	AINL,
	AINW,
	AINCB,
	AINCL,
	AINCW,
	AINSB,
	AINSL,
	AINSW,
	AINT,
	AINTO,
	AIRETL,
	AIRETW,
	AJCC,
	AJCS,
	AJCXZ,
	AJEQ,
	AJGE,
	AJGT,
	AJHI,
	AJLE,
	AJLS,
	AJLT,
	AJMI,
	AJMP,
	AJNE,
	AJOC,
	AJOS,
	AJPC,
	AJPL,
	AJPS,
	ALAHF,
	ALARL,
	ALARW,
	ALEAL,
	ALEAW,
	ALEAVEL,
	ALEAVEW,
	ALOCK,
	ALODSB,
	ALODSL,
	ALODSW,
	ALONG,
	ALOOP,
	ALOOPEQ,
	ALOOPNE,
	ALSLL,
	ALSLW,
	AMOVB,
	AMOVL,
	AMOVW,
	AMOVBLSX,
	AMOVBLZX,
	AMOVBWSX,
	AMOVBWZX,
	AMOVWLSX,
	AMOVWLZX,
	AMOVSB,
	AMOVSL,
	AMOVSW,
	AMULB,
	AMULL,
	AMULW,
	ANAME,
	ANEGB,
	ANEGL,
	ANEGW,
	ANOP,
	ANOTB,
	ANOTL,
	ANOTW,
	AORB,
	AORL,
	AORW,
	AOUTB,
	AOUTL,
	AOUTW,
	AOUTSB,
	AOUTSL,
	AOUTSW,
	APOPAL,
	APOPAW,
	APOPFL,
	APOPFW,
	APOPL,
	APOPW,
	APUSHAL,
	APUSHAW,
	APUSHFL,
	APUSHFW,
	APUSHL,
	APUSHW,
	ARCLB,
	ARCLL,
	ARCLW,
	ARCRB,
	ARCRL,
	ARCRW,
	AREP,
	AREPN,
	ARET,
	AROLB,
	AROLL,
	AROLW,
	ARORB,
	ARORL,
	ARORW,
	ASAHF,
	ASALB,
	ASALL,
	ASALW,
	ASARB,
	ASARL,
	ASARW,
	ASBBB,
	ASBBL,
	ASBBW,
	ASCASB,
	ASCASL,
	ASCASW,
	ASETCC,
	ASETCS,
	ASETEQ,
	ASETGE,
	ASETGT,
	ASETHI,
	ASETLE,
	ASETLS,
	ASETLT,
	ASETMI,
	ASETNE,
	ASETOC,
	ASETOS,
	ASETPC,
	ASETPL,
	ASETPS,
	ACDQ,
	ACWD,
	ASHLB,
	ASHLL,
	ASHLW,
	ASHRB,
	ASHRL,
	ASHRW,
	ASTC,
	ASTD,
	ASTI,
	ASTOSB,
	ASTOSL,
	ASTOSW,
	ASUBB,
	ASUBL,
	ASUBW,
	ASYSCALL,
	ATESTB,
	ATESTL,
	ATESTW,
	ATEXT,
	AVERR,
	AVERW,
	AWAIT,
	AWORD,
	AXCHGB,
	AXCHGL,
	AXCHGW,
	AXLAT,
	AXORB,
	AXORL,
	AXORW,

	AFMOVB,
	AFMOVBP,
	AFMOVD,
	AFMOVDP,
	AFMOVF,
	AFMOVFP,
	AFMOVL,
	AFMOVLP,
	AFMOVV,
	AFMOVVP,
	AFMOVW,
	AFMOVWP,
	AFMOVX,
	AFMOVXP,

	AFCOMB,
	AFCOMBP,
	AFCOMD,
	AFCOMDP,
	AFCOMDPP,
	AFCOMF,
	AFCOMFP,
	AFCOMI,
	AFCOMIP,
	AFCOML,
	AFCOMLP,
	AFCOMW,
	AFCOMWP,
	AFUCOM,
	AFUCOMI,
	AFUCOMIP,
	AFUCOMP,
	AFUCOMPP,

	AFADDDP,
	AFADDW,
	AFADDL,
	AFADDF,
	AFADDD,

	AFMULDP,
	AFMULW,
	AFMULL,
	AFMULF,
	AFMULD,

	AFSUBDP,
	AFSUBW,
	AFSUBL,
	AFSUBF,
	AFSUBD,

	AFSUBRDP,
	AFSUBRW,
	AFSUBRL,
	AFSUBRF,
	AFSUBRD,

	AFDIVDP,
	AFDIVW,
	AFDIVL,
	AFDIVF,
	AFDIVD,

	AFDIVRDP,
	AFDIVRW,
	AFDIVRL,
	AFDIVRF,
	AFDIVRD,

	AFXCHD,
	AFFREE,

	AFLDCW,
	AFLDENV,
	AFRSTOR,
	AFSAVE,
	AFSTCW,
	AFSTENV,
	AFSTSW,

	AF2XM1,
	AFABS,
	AFCHS,
	AFCLEX,
	AFCOS,
	AFDECSTP,
	AFINCSTP,
	AFINIT,
	AFLD1,
	AFLDL2E,
	AFLDL2T,
	AFLDLG2,
	AFLDLN2,
	AFLDPI,
	AFLDZ,
	AFNOP,
	AFPATAN,
	AFPREM,
	AFPREM1,
	AFPTAN,
	AFRNDINT,
	AFSCALE,
	AFSIN,
	AFSINCOS,
	AFSQRT,
	AFTST,
	AFXAM,
	AFXTRACT,
	AFYL2X,
	AFYL2XP1,

	AEND,

	ADYNT,
	AINIT,

	ASIGNAME,

	ACMPXCHGB,
	ACMPXCHGL,
	ACMPXCHGW,

	/* conditional move */
	ACMOVLCC,
	ACMOVLCS,
	ACMOVLEQ,
	ACMOVLGE,
	ACMOVLGT,
	ACMOVLHI,
	ACMOVLLE,
	ACMOVLLS,
	ACMOVLLT,
	ACMOVLMI,
	ACMOVLNE,
	ACMOVLOC,
	ACMOVLOS,
	ACMOVLPC,
	ACMOVLPL,
	ACMOVLPS,
	ACMOVWCC,
	ACMOVWCS,
	ACMOVWEQ,
	ACMOVWGE,
	ACMOVWGT,
	ACMOVWHI,
	ACMOVWLE,
	ACMOVWLS,
	ACMOVWLT,
	ACMOVWMI,
	ACMOVWNE,
	ACMOVWOC,
	ACMOVWOS,
	ACMOVWPC,
	ACMOVWPL,
	ACMOVWPS,

	AFCMOVCC,
	AFCMOVCS,
	AFCMOVEQ,
	AFCMOVHI,
	AFCMOVLS,
	AFCMOVNE,
	AFCMOVNU,
	AFCMOVUN,

	ALAST
};

enum
{
	D_AL		= 0,
	D_CL,
	D_DL,
	D_BL,

	D_AH		= 4,
	D_CH,
	D_DH,
	D_BH,

	D_AX		= 8,
	D_CX,
	D_DX,
	D_BX,
	D_SP,
	D_BP,
	D_SI,
	D_DI,

	D_F0		= 16,
	D_F7		= D_F0 + 7,

	D_CS		= 24,
	D_SS,
	D_DS,
	D_ES,
	D_FS,
	D_GS,

	D_GDTR,		/* global descriptor table register */
	D_IDTR,		/* interrupt descriptor table register */
	D_LDTR,		/* local descriptor table register */
	D_MSW,		/* machine status word */
	D_TASK,		/* task register */

	D_CR		= 35,
	D_DR		= 43,
	D_TR		= 51,

	D_NONE		= 59,

	D_BRANCH	= 60,
	D_EXTERN	= 61,
	D_STATIC	= 62,
	D_AUTO		= 63,
	D_PARAM		= 64,
	D_CONST		= 65,
	D_FCONST	= 66,
	D_SCONST	= 67,
	D_ADDR		= 68,

	D_FILE,
	D_FILE1,

	D_INDIR,	/* additive */

	D_CONST2 = D_INDIR+D_INDIR,
	D_SIZE,	/* 8l internal */

	T_TYPE		= 1<<0,
	T_INDEX		= 1<<1,
	T_OFFSET	= 1<<2,
	T_FCONST	= 1<<3,
	T_SYM		= 1<<4,
	T_SCONST	= 1<<5,
	T_OFFSET2	= 1<<6,
	T_GOTYPE	= 1<<7,

	REGARG		= -1,
	REGRET		= D_AX,
	FREGRET		= D_F0,
	REGSP		= D_SP,
	REGTMP		= D_DI,
};

/*
 * this is the ranlib header
 */
#define	SYMDEF	"__.SYMDEF"

/*
 * this is the simulated IEEE floating point
 */
typedef	struct	ieee	Ieee;
struct	ieee
{
	int32	l;	/* contains ls-man	0xffffffff */
	int32	h;	/* contains sign	0x80000000
				    exp		0x7ff00000
				    ms-man	0x000fffff */
};