
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_airy_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_airy", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.976316202008158e+01, -1.976320417836311e+01, -1.976343599356560e+01, -1.976274478792864e+01, -1.976318388730053e+01, -1.976318388730053e+01, -3.236095007261377e+00, -3.236088585294406e+00, -3.236046493091491e+00, -3.236806384493631e+00, -3.236105489301795e+00, -3.236105489301795e+00, -6.355433463080999e-01, -6.353119154504633e-01, -6.309043445544192e-01, -6.351363188694490e-01, -6.354566668812742e-01, -6.354566668812742e-01, -1.896253744500765e-01, -1.907849960417685e-01, -7.495068704802880e-01, -1.574262017627857e-01, -1.899535278035100e-01, -1.899535278035100e-01, -4.255052154222425e-02, -4.312434429239941e-02, -8.522906871180885e-02, -3.794002261893051e-02, -4.270326905509492e-02, -4.270326905509492e-02, -4.805062871136026e+00, -4.805807075458116e+00, -4.805141785730267e+00, -4.805720359999862e+00, -4.805439653508977e+00, -4.805439653508977e+00, -1.900981701268079e+00, -1.911271387852989e+00, -1.899959097822552e+00, -1.907961073511390e+00, -1.909327941250933e+00, -1.909327941250933e+00, -5.381927003863022e-01, -5.827519293769474e-01, -5.113416995564768e-01, -5.303126571583363e-01, -5.577304926180067e-01, -5.577304926180067e-01, -1.350026292474953e-01, -2.057686432963777e-01, -1.317327108058850e-01, -1.772491211000105e+00, -1.424952070353616e-01, -1.424952070353616e-01, -3.722550057664015e-02, -3.824508944759980e-02, -2.924660450772505e-02, -1.001181860010282e-01, -3.472466595298131e-02, -3.472466595298133e-02, -5.551244096103419e-01, -5.508422707008495e-01, -5.523356727994141e-01, -5.535264145603408e-01, -5.529271889572303e-01, -5.529271889572305e-01, -5.375331491929416e-01, -4.742323078189384e-01, -4.883193620574641e-01, -5.041983666520111e-01, -4.956244523098250e-01, -4.956244523098251e-01, -6.108484548340304e-01, -2.453141087329476e-01, -2.780823854004451e-01, -3.340747358482570e-01, -3.037851827072377e-01, -3.037851827072335e-01, -4.292259525200810e-01, -8.412023431852420e-02, -9.702459093054019e-02, -3.106462709452934e-01, -1.141225090002665e-01, -1.141225090002672e-01, -4.692378409593700e-02, -2.742909974365934e-02, -3.122222373551187e-02, -1.086279892664861e-01, -3.114233939676346e-02, -3.114233939676345e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_airy_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_airy", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.575085293801162e+01, -2.575095403039068e+01, -2.575139525775735e+01, -2.574973999470667e+01, -2.575090633238267e+01, -2.575090633238267e+01, -4.100416941394788e+00, -4.100477362723306e+00, -4.102307401125270e+00, -4.100023908230094e+00, -4.100465484913691e+00, -4.100465484913691e+00, -7.513426022507200e-01, -7.500320822313130e-01, -7.188868042863780e-01, -7.250141047662202e-01, -7.508659135590363e-01, -7.508659135590363e-01, -1.970450026788849e-01, -2.002471639544887e-01, -9.229019366643817e-01, -1.361008377606679e-01, -1.980155555723142e-01, -1.980155555723142e-01, -1.700590402243909e-02, -1.764192682549546e-02, -5.712890844379471e-02, -1.022813744603092e-02, -1.746144007736222e-02, -1.746144007736222e-02, -6.368007819934256e+00, -6.370890406741274e+00, -6.368302851351304e+00, -6.370544364241375e+00, -6.369488384423361e+00, -6.369488384423361e+00, -2.169405950046325e+00, -2.186740172540055e+00, -2.160494784074189e+00, -2.173926315263198e+00, -2.193057200200518e+00, -2.193057200200518e+00, -6.925811771327021e-01, -7.817801710379051e-01, -6.540642753037071e-01, -7.100859400228955e-01, -7.256442863652930e-01, -7.256442863652930e-01, -1.035721840352990e-01, -1.921922724542609e-01, -1.016365508755773e-01, -2.377858353178761e+00, -1.157900307810954e-01, -1.157900307810954e-01, -9.942300215556837e-03, -1.084661989537658e-02, -8.199132481901659e-03, -7.217975332299044e-02, -9.868320314306348e-03, -9.868320314306447e-03, -7.420385360987266e-01, -7.388658988103961e-01, -7.403203066383381e-01, -7.411916819609303e-01, -7.407820770830518e-01, -7.407820770830518e-01, -7.178708592584317e-01, -5.751601405108308e-01, -6.210783284749770e-01, -6.664942703770004e-01, -6.439328536553887e-01, -6.439328536553861e-01, -8.194500923460236e-01, -2.473567495149619e-01, -2.993934316944939e-01, -3.957834614304405e-01, -3.430128170472899e-01, -3.430128170472959e-01, -5.159427420276521e-01, -5.444973999814567e-02, -6.802470420522520e-02, -3.835129414725478e-01, -8.606491890043680e-02, -8.606491890043541e-02, -1.958151101858569e-02, -5.091252932026478e-03, -7.024644748194172e-03, -8.199255773677279e-02, -8.581967257403185e-03, -8.581967257403143e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_airy_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_airy", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.519285462923223e-09, -1.519192635857637e-09, -1.518880562692234e-09, -1.520398562878065e-09, -1.519235656132540e-09, -1.519235656132540e-09, -3.431030186202729e-06, -3.430454020017190e-06, -3.413991501973983e-06, -3.439364298636768e-06, -3.430675412369778e-06, -3.430675412369778e-06, -3.266835856725686e-03, -3.279536536601217e-03, -3.477897183373680e-03, -3.384621068769185e-03, -3.271518838561073e-03, -3.271518838561073e-03, -4.343103015879861e-01, -4.212805306441687e-01, -1.476553586604707e-03, -1.109530474031878e+00, -4.304376465399655e-01, -4.304376465399655e-01, -1.773771681178430e+03, -1.551425438062557e+03, -2.251739976748103e+01, -1.367275251532773e+04, -1.616130430681389e+03, -1.616130430681389e+03, -1.569434201484588e-07, -1.503192865537033e-07, -1.562811328130765e-07, -1.511303438845394e-07, -1.535266758208818e-07, -1.535266758208818e-07, -4.217908518804330e-05, -4.124664926214360e-05, -4.230299126771763e-05, -4.158125698731549e-05, -4.135145947063323e-05, -4.135145947063323e-05, -3.644887580302440e-03, 1.804537852646894e-03, -4.909548988541524e-03, 1.160867276890840e-03, -2.496434151229870e-03, -2.496434151229870e-03, -2.443806115323178e+00, -3.469565987219030e-01, -2.666489060392652e+00, 2.093141403476947e-05, -1.792760230624591e+00, -1.792760230624591e+00, -1.542651361711614e+04, -1.046778257176980e+04, -3.226257943579919e+04, -9.514054902671537e+00, -1.526171663692010e+04, -1.526171663692010e+04, 3.171417452135717e-03, 2.517336298531318e-03, 2.889218226766984e-03, 3.099073995262781e-03, 3.005302354246314e-03, 3.005302354245950e-03, 3.480254892737480e-03, -9.845399525980424e-03, -6.346693436568589e-03, -1.729079874124828e-03, -4.154074618897051e-03, -4.154074618896363e-03, 1.359991138391100e-03, -1.587374750976153e-01, -9.239273713860993e-02, -4.262082870386191e-02, -6.477520248646031e-02, -6.477520248646682e-02, -1.510562473080695e-02, -2.623978397062287e+01, -1.174423909151895e+01, -4.934071228998550e-02, -4.965200442782626e+00, -4.965200442782569e+00, -1.034426748570070e+03, -5.239154416684775e+05, -7.798092108030601e+04, -6.036347162845055e+00, -2.716292625512289e+04, -2.716292625512308e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05