
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_bop_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_bop", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.082979447170749e+01, -2.082981859309688e+01, -2.083000289103880e+01, -2.082960636451173e+01, -2.082980655148262e+01, -2.082980655148262e+01, -3.255262447403572e+00, -3.255238932967480e+00, -3.254736616692578e+00, -3.256342043762955e+00, -3.255265213899651e+00, -3.255265213899651e+00, -4.665128523946693e-01, -4.660911500679883e-01, -4.556969318949303e-01, -4.604712348799678e-01, -4.663578090825219e-01, -4.663578090825219e-01, -4.682767805440298e-02, -4.835279105985494e-02, -5.893746845956895e-01, -2.157988783562644e-02, -4.728644560558797e-02, -4.728644560558797e-02, -4.407829914357134e-05, -4.964306259743166e-05, -1.353873236853134e-03, -5.820008201221737e-06, -4.894733452324793e-05, -4.894733452324793e-05, -4.837809211266685e+00, -4.837314589833454e+00, -4.837764758695464e+00, -4.837380063501833e+00, -4.837546854514683e+00, -4.837546854514683e+00, -1.845949061399197e+00, -1.856881104948521e+00, -1.845480243176270e+00, -1.853998593587017e+00, -1.853969047381313e+00, -1.853969047381313e+00, -3.728496281457326e-01, -4.113574065104445e-01, -3.458721086597699e-01, -3.622775016465165e-01, -3.924329350700418e-01, -3.924329350700418e-01, -9.840222217352475e-03, -4.541496310604175e-02, -9.696632671635756e-03, -1.614949130912190e+00, -1.451736118575613e-02, -1.451736118575613e-02, -5.409547072547928e-06, -7.690953634417838e-06, -5.349186997097342e-06, -3.152145884012790e-03, -6.950086773267523e-06, -6.950086773267523e-06, -3.809844600669369e-01, -3.797354896434780e-01, -3.801957119598642e-01, -3.805406109160451e-01, -3.803687881037724e-01, -3.803687881037724e-01, -3.637031333364675e-01, -3.051755759104794e-01, -3.227906337585645e-01, -3.386994801724443e-01, -3.306061200768798e-01, -3.306061200768798e-01, -4.393803428036018e-01, -7.540468210339440e-02, -1.076713042815292e-01, -1.697327559421571e-01, -1.363239685287606e-01, -1.363239685287606e-01, -2.600892386593191e-01, -1.074297942467202e-03, -2.334943764539930e-03, -1.537169471632996e-01, -6.201760267252308e-03, -6.201760267252353e-03, -6.019979144750073e-05, -2.998073196141569e-07, -1.582670831012398e-06, -5.690105270081238e-03, -5.232304661892673e-06, -5.232304661892653e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_bop_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_bop", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.506323296961710e+01, -2.506331471800213e+01, -2.506369476732929e+01, -2.506235581340364e+01, -2.506327595260284e+01, -2.506327595260284e+01, -3.917875590697519e+00, -3.917903646953940e+00, -3.918842313011743e+00, -3.918086109965514e+00, -3.917907320238805e+00, -3.917907320238805e+00, -6.282129898106501e-01, -6.273720166151652e-01, -6.059227337478379e-01, -6.117983589669552e-01, -6.279063897685565e-01, -6.279063897685565e-01, -8.863338919380534e-02, -9.101970424805493e-02, -7.836881931505566e-01, -4.516168023458157e-02, -8.935189511355995e-02, -8.935189511355995e-02, -1.393932265148869e-04, -1.563298690302785e-04, -3.548890558245623e-03, -1.941353719551929e-05, -1.543161320362807e-04, -1.543161320362807e-04, -6.044330832175079e+00, -6.046580948170114e+00, -6.044561754110466e+00, -6.046311318559735e+00, -6.045483171760469e+00, -6.045483171760469e+00, -2.087526198549218e+00, -2.104318493575534e+00, -2.079544940034756e+00, -2.092615560862590e+00, -2.109352623726772e+00, -2.109352623726772e+00, -5.307737380383454e-01, -5.930120129735396e-01, -4.959412069242723e-01, -5.254178004340774e-01, -5.573079512781618e-01, -5.573079512781618e-01, -2.218147314857224e-02, -8.716174263081319e-02, -2.191972505296138e-02, -2.152825568407967e+00, -3.164428606081602e-02, -3.164428606081602e-02, -1.809378964845475e-05, -2.550726281676247e-05, -1.812624421558008e-05, -7.812933551541250e-03, -2.324581475297005e-05, -2.324581475297005e-05, -5.560936366351186e-01, -5.514786809587621e-01, -5.531135634853154e-01, -5.543890765454990e-01, -5.537491617559166e-01, -5.537491617559166e-01, -5.330167458498006e-01, -4.397478389471819e-01, -4.661391121635448e-01, -4.911514727652260e-01, -4.783450476844036e-01, -4.783450476844036e-01, -6.295644144814675e-01, -1.346769106678767e-01, -1.815109968019209e-01, -2.659001294690494e-01, -2.209529287984205e-01, -2.209529287984205e-01, -3.829358338234963e-01, -2.843266322028467e-03, -5.886287206906796e-03, -2.437878775978994e-01, -1.460975071783725e-02, -1.460975071783704e-02, -1.872117800329481e-04, -1.061413079542477e-06, -5.458132612992177e-06, -1.353284115757182e-02, -1.769385703378105e-05, -1.769385703378098e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_bop_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_bop", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.055388499696396e-09, -7.055358938388613e-09, -7.055121402694739e-09, -7.055607605067583e-09, -7.055373793255429e-09, -7.055373793255429e-09, -8.006023106220358e-06, -8.006273696754880e-06, -8.011907947828338e-06, -7.995740462514924e-06, -8.006021947568646e-06, -8.006021947568646e-06, -2.007737914810858e-03, -2.007318060815944e-03, -1.982949183761533e-03, -1.952181953570225e-03, -2.007612746363418e-03, -2.007612746363418e-03, 7.836662483099642e-02, 7.818173710535892e-02, -1.320306017209341e-03, 1.274420605039170e-01, 7.838490151168716e-02, 7.838490151168716e-02, 3.452349731860874e+00, 3.360099244243460e+00, 5.682108792432264e-01, 3.832366680225972e+00, 3.495264206189605e+00, 3.495264206189605e+00, -1.939225051183465e-06, -1.940543674331253e-06, -1.939349163276244e-06, -1.940374577973169e-06, -1.939915358635653e-06, -1.939915358635653e-06, -5.334699767322471e-05, -5.239408852356460e-05, -5.332265306470735e-05, -5.257852896342651e-05, -5.273340247324588e-05, -5.273340247324588e-05, -3.196837295061738e-03, -3.108861280372312e-03, -3.494171884366483e-03, -3.598203768121823e-03, -3.016202542111337e-03, -3.016202542111337e-03, 1.776576720447752e-01, 3.878151465678185e-02, 2.028670344249476e-01, -9.466384651855656e-05, 1.719818422968786e-01, 1.719818422968786e-01, 4.111536758729519e+00, 3.857036063067475e+00, 1.135813100383483e+01, 4.256284148756070e-01, 5.720691984831720e+00, 5.720691984831721e+00, -3.669907698156081e-03, -3.513725503411590e-03, -3.559825548819516e-03, -3.601856916489923e-03, -3.580020412833005e-03, -3.580020412833005e-03, -3.928038985419826e-03, -3.775520885590343e-03, -3.745158611897497e-03, -3.739848477651982e-03, -3.740820078168142e-03, -3.740820078168142e-03, -2.779084719848456e-03, 1.447143150805402e-02, 5.079437477070015e-03, -1.947289175098472e-03, 9.433354264633820e-04, 9.433354264634201e-04, -4.147150504231284e-03, 5.262900531193914e-01, 4.092952555544829e-01, 1.463633312520151e-03, 3.338698025346760e-01, 3.338698025346575e-01, 2.439044461469074e+00, 1.089912740968452e+01, 7.421672774456283e+00, 4.136590445275777e-01, 8.689443087209257e+00, 8.689443087209259e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05