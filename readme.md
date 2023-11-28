# HOPSFlow Paper Code

These are the stripped down and cleaned bare bones of the code used to
obtain most of the results in the HOPS otto-engine paper [arxiv link
pending]. This is the raw-in-the-fire code that I'm not extremely
proud of (formatting, lack of docs). The python modules mentioned
below are significantly nicer in that regard.

It uses our HOPS implementation [code publication pending?], as well
as <https://github.com/vale981/two_qubit_model>,
<https://github.com/vale981/HOPSFlow> and <https://github.com/vale981/stocproc>.

Most of the code uses literate org-mode files called `project.org` org
`cycle_shift.org` in one occasion. The `subprojects` directory holds the actual code used in the paper:
 - `subprojects/cycle_shift` has the code for the shifted otto cycles
 - `subprojects/cycle_length_coupling_strength` contains the code for
   the shifted scan of coupling strength and cycle length
 - the other folder contain experiments that didn't make it into the
   paper (which is already too long as it stands)

It takes quite a lot to get the code to actually run, even on
`nixos`. If you really want to embark on that route please contact me
if you encounter trouble. I may not have much time to help you though.

The actual simulation data is available upon request. Checking
gigabytes of binary data into git is not a good idea.u
