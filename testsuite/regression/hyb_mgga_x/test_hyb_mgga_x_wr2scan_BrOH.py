
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_wr2scan_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.298284686358916e+01, -2.298290693358751e+01, -2.298324833696620e+01, -2.298226605238008e+01, -2.298287789214428e+01, -2.298287789214428e+01, -3.223543987562236e+00, -3.223656474582715e+00, -3.227301130167318e+00, -3.228203580766955e+00, -3.223578230149760e+00, -3.223578230149760e+00, -4.443077690137315e-01, -4.437173037534450e-01, -4.291014737449992e-01, -4.378430251199593e-01, -4.440978796995051e-01, -4.440978796995051e-01, -6.307032200611765e-02, -6.461807095600650e-02, -5.241837349630231e-01, -2.530078850248826e-02, -6.349178988446766e-02, -6.349178988446766e-02, -1.074411311677136e-05, -1.248923628557274e-05, -1.064936445731626e-03, -8.991170858535728e-07, -1.211899113227540e-05, -1.211899113227540e-05, -5.469407417249891e+00, -5.470475384584716e+00, -5.469533634231481e+00, -5.470363167515454e+00, -5.469932087660271e+00, -5.469932087660271e+00, -1.887060228803120e+00, -1.908825294520496e+00, -1.884399475748543e+00, -1.901548717731497e+00, -1.905408316365089e+00, -1.905408316365089e+00, -4.355983732354805e-01, -4.854772416098500e-01, -3.922370759511940e-01, -4.144539005940407e-01, -4.637504790474700e-01, -4.637504790474700e-01, -1.136435825548869e-02, -6.277405657739681e-02, -1.103259732733500e-02, -1.734896442329328e+00, -1.749775769982755e-02, -1.749775769982755e-02, -8.014586855507744e-07, -1.253557990194241e-06, -5.236536408810744e-07, -2.910436582170432e-03, -9.497205631901253e-07, -9.497205631901253e-07, -4.732393839768120e-01, -4.699190507588435e-01, -4.711091536966566e-01, -4.720278048052776e-01, -4.715670456533439e-01, -4.715670456533439e-01, -4.510090130682860e-01, -3.716453532665246e-01, -3.948761678798372e-01, -4.163730189073869e-01, -4.052841981855779e-01, -4.052841981855779e-01, -5.081754746554901e-01, -1.029097747535863e-01, -1.389528407622821e-01, -2.037857056794562e-01, -1.707782710201937e-01, -1.707782710201937e-01, -3.110328390438598e-01, -8.367695270411833e-04, -2.115064886320765e-03, -1.864560084750968e-01, -6.459951271700220e-03, -6.459951271700220e-03, -1.752936362180886e-05, -1.654386884629132e-08, -1.442082257323532e-07, -5.668029010704985e-03, -5.693639227875135e-07, -5.693639227875105e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_wr2scan_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.995484826331173e+01, -2.995494225563256e+01, -2.995540578678399e+01, -2.995386729381205e+01, -2.995489744215549e+01, -2.995489744215549e+01, -4.617868833784797e+00, -4.617965566162837e+00, -4.620944105642224e+00, -4.619112476350024e+00, -4.617925228450629e+00, -4.617925228450629e+00, -6.614713626403557e-01, -6.604562422295726e-01, -6.350077411426073e-01, -6.450728185852048e-01, -6.611071680678552e-01, -6.611071680678552e-01, -8.988586161955463e-02, -9.321958555394319e-02, -7.394492851632155e-01, -4.636337272269450e-02, -9.083405876445599e-02, -9.083405876445599e-02, -1.926504140658862e-05, -2.502851140424220e-05, -2.280745108561826e-03, -2.222087552428598e-06, -2.265273393738086e-05, -2.265273393738106e-05, -7.210328110737797e+00, -7.212637081773586e+00, -7.210572332689115e+00, -7.212367275254866e+00, -7.211501359435495e+00, -7.211501359435495e+00, -2.368856785110081e+00, -2.403039196151992e+00, -2.359272502095152e+00, -2.387018704089848e+00, -2.404756431495543e+00, -2.404756431495543e+00, -6.335093531603481e-01, -7.307997522598147e-01, -5.906636692963599e-01, -6.454547996839263e-01, -6.679204750102808e-01, -6.679204750102808e-01, -2.158407214533404e-02, -7.294020478936249e-02, -2.088124681706150e-02, -2.625401711884017e+00, -2.795965326037908e-02, -2.795965326037908e-02, -1.979890660444372e-06, -3.084694520765271e-06, -1.285888053331125e-06, -6.141426266514236e-03, -2.337415236364809e-06, -2.337415236364809e-06, -6.841939047167740e-01, -6.756446930512824e-01, -6.786947289687290e-01, -6.810646022972835e-01, -6.798783298303740e-01, -6.798783298303740e-01, -6.581425703303326e-01, -5.212920430164214e-01, -5.599983466271597e-01, -5.963424084108765e-01, -5.778245864171638e-01, -5.778245864171638e-01, -7.763078944821304e-01, -1.363223149608029e-01, -1.916690810201165e-01, -3.059942675442128e-01, -2.453610103390146e-01, -2.453610103390146e-01, -4.442171076652429e-01, -1.719055917906720e-03, -4.451328608367998e-03, -2.894488786282668e-01, -1.234021704456402e-02, -1.234021704456401e-02, -4.167944457607733e-05, -4.211562625801648e-08, -3.592408963480505e-07, -1.099816241741141e-02, -1.404831368307699e-06, -1.404831368307692e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_wr2scan_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.243241795891193e-09, -6.243189419416796e-09, -6.242704349694746e-09, -6.243552237961344e-09, -6.243216466600329e-09, -6.243216466600329e-09, -1.440372485470915e-05, -1.440519499419737e-05, -1.444647230035069e-05, -1.442495191162130e-05, -1.440386625304691e-05, -1.440386625304691e-05, -3.198016229815791e-03, -3.188943517004882e-03, -2.984169987441720e-03, -3.104005019331432e-03, -3.194971000035473e-03, -3.194971000035473e-03, -5.023373538405170e-01, -4.949038646449013e-01, -1.152563742300828e-03, -6.351746577017717e-02, -5.016579087502593e-01, -5.016579087502593e-01, -5.879501120249168e-01, -3.159211655719548e-01, 1.606880332317333e-02, 1.431981736946214e-01, -5.028190262346688e-01, -5.028190262346458e-01, -1.715907748578466e-06, -1.716585783479947e-06, -1.715882800203362e-06, -1.716415065640808e-06, -1.716380380524374e-06, -1.716380380524374e-06, -9.226923804399533e-05, -8.573898882287173e-05, -9.055676274675418e-05, -8.521597656182627e-05, -8.990667936713971e-05, -8.990667936713971e-05, -1.335198509277480e-02, -1.174992442536413e-02, -1.779423605620053e-02, -1.927698270409406e-02, -1.086458023652621e-02, -1.086458023652621e-02, -9.595780982685785e-02, -4.775183712268195e-01, -1.162275644510448e-01, -1.964582291591784e-04, -4.844003589321585e-01, -4.844003589321585e-01, 1.445293217795182e-01, 1.497645610342807e-01, 2.411821902167114e-01, 5.600139882905969e-02, 1.829515348006151e-01, 1.829515348006150e-01, -8.978835626108078e-03, -9.008072866267385e-03, -9.016911795740499e-03, -9.007242906499773e-03, -9.013931054137937e-03, -9.013931054137938e-03, -1.079438015472481e-02, -1.326660076121019e-02, -1.229593387047414e-02, -1.176986060229626e-02, -1.205387310854513e-02, -1.205387310854512e-02, -1.077029086698192e-02, -2.037443688255527e-01, -1.435177305474614e-01, -7.730353616535209e-02, -1.039766287392361e-01, -1.039766287392361e-01, -2.842336139733014e-02, -9.740671580340439e-02, 8.847158397448021e-03, -9.975921295558250e-02, -1.780394653609396e-01, -1.780394653609378e-01, 1.589201817648416e-01, 1.500411544332174e-01, 1.506665362837225e-01, -1.585036051603837e-01, 2.188505364314962e-01, 2.188505364314959e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_wr2scan_BrOH_1_vlapl():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_wr2scan_BrOH_1_vtau():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_wr2scan", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.003292445595373e-03, 1.003290846460207e-03, 1.003254807780730e-03, 1.003279599683826e-03, 1.003291869239285e-03, 1.003291869239285e-03, 8.752454319650082e-03, 8.753669637598651e-03, 8.790219215094595e-03, 8.787191020355500e-03, 8.752674230636050e-03, 8.752674230636050e-03, 1.119170606262041e-02, 1.116249850500344e-02, 1.057531033103649e-02, 1.152932140195539e-02, 1.118230802664922e-02, 1.118230802664922e-02, 4.990478809716179e-02, 5.103043280266904e-02, 2.520091101068419e-03, 3.086062550413643e-03, 5.038598836121806e-02, 5.038598836121806e-02, 1.286892616765991e-05, 9.579740893171559e-06, 1.021436270965332e-04, 1.063959946772159e-09, 1.282300228487414e-05, 1.282300228487372e-05, 3.988148398443928e-03, 3.990988703502040e-03, 3.988275716947893e-03, 3.990493238100135e-03, 3.989819832504093e-03, 3.989819832504093e-03, 1.135963593124728e-02, 1.082590430764490e-02, 1.108830290652631e-02, 1.064617976447371e-02, 1.133786864407021e-02, 1.133786864407021e-02, 4.015466676949473e-02, 4.275787549450195e-02, 4.417325751230870e-02, 4.801458824411134e-02, 3.684623981466421e-02, 3.684623981466421e-02, 1.991903367735575e-03, 4.844711994483444e-02, 2.231463863473173e-03, 1.878116411749011e-02, 1.168276764894919e-02, 1.168276764894919e-02, 4.646882393727105e-09, 2.973073052094406e-09, 1.318846610566793e-08, 7.624098477096188e-05, 1.419259462665463e-09, 1.419259462665457e-09, 3.061608026279992e-02, 3.027592465609449e-02, 3.047273250214339e-02, 3.056477885099852e-02, 3.052600977176272e-02, 3.052600977176272e-02, 3.331756900655156e-02, 2.820659602075042e-02, 2.935671900077859e-02, 3.076507140658551e-02, 3.015201626255238e-02, 3.015201626255237e-02, 4.359990758663047e-02, 4.169441130884954e-02, 4.849579242954220e-02, 5.205255226157941e-02, 5.013009131471862e-02, 5.013009131471861e-02, 4.235259454105505e-02, 1.940625162076861e-04, 1.789925903750947e-04, 5.553655271107005e-02, 1.862971296105336e-03, 1.862971296105323e-03, 4.318558567498993e-08, 1.057503465901766e-11, 4.592422961548812e-09, 1.533564591849169e-03, 8.897335539376984e-10, 8.897335539376946e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05